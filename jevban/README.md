# jevban: fail2ban for web attackers, judged by a Jev-style decision AI

A Free Pascal command-line daemon. It follows Apache access logs, spots clients that are
hunting for security holes, and bans them in the firewall, the way fail2ban does. The
difference from a fail2ban filter: a regex is only the **prefilter**. A **decision model**
(OpenJev, or the hosted Jev API) decides whether a client is attacking. It sees the client's
recent requests and a plain-words description of your site.

```
access.log ──tail -F──▶ parse ──▶ signatures (cheap) ──suspicious──▶ POST /v1/systemone
                                                                         │ P(malicious), intent
                     firewall ◀── ban ◀── maxretry strikes in findtime ◀──┘
```

Why a model: `GET /wp-login.php` is a probe on a static site and a normal login on a WordPress
site. You write that in `site =` and the model weighs it. A regex can't.

## Build and try it

```bash
make jevban            # -> build/jevban   (FPC 3.2.2; no other dependencies for http://)
make test-jevban       # unit tests + a replay of jevban/tests/sample_access.log
```

```bash
./build/jevban --test /var/log/apache2/access.log --mode rules       # what would be banned, bans nothing
./build/jevban --test /var/log/apache2/access.log                    # the same, asking the decision API
python3 jevban/tests/mock_jev.py 3000 &                              # a fake decider, no GPU needed
```

A run over the sample log (`--mode rules`):

```
203.0.113.66 strike 1/3 score=10 [env-file] via rules: "GET /.env HTTP/1.1" 404 ...
203.0.113.66 strike 2/3 score=10 [vcs-dir] via rules: "GET /.git/config HTTP/1.1" 404 ...
203.0.113.66 strike 3/3 score=9 [config-file,backup-file] via rules: "GET /wp-config.php.bak HTTP/1.1" 404 ...
Ban 203.0.113.66 for 3600s (ban #1): config-file,backup-file, score 9
192.0.2.44 BAN score=20 [sqli,scanner-agent] via rules: "GET /?id=1%20UNION%20SELECT..." ua="sqlmap/1.8..."
198.51.100.7 pass score=3 [wordpress] via rules: "GET /wp-login.php HTTP/1.1" 200 ...
15 lines, 15 parsed, 1 ignored, 9 suspicious, 0 decider calls (0 failed), 8 strikes, 2 bans, 0 unbans
```

## Run it for real

```bash
sudo install -m 755 build/jevban /usr/local/bin/
sudo install -D -m 644 jevban/jevban.ini /etc/jevban/jevban.ini     # edit logpath, site, action
sudo jevban --dry-run                                                # watch the firewall commands it would run
sudo install -m 644 jevban/jevban.service /etc/systemd/system/ && sudo systemctl enable --now jevban
sudo jevban --list                                                   # active bans
```

It has to run as root (or with `CAP_NET_ADMIN`) to change the firewall.

## The decider (Jev compatible)

`mode = jev` (the default) sends one `POST /v1/systemone` per suspicious client. That is
the request shape of the hosted Jev API, which [OpenJev](https://huggingface.co/openjev/openjev)'s
helper (`helper/shim.py`, port 3000 in its SERVE.md) serves on top of vLLM. Point `url`
at either one. Set `api_key` or `JEV_API_KEY` if the server wants a bearer token.

```json
{"model": "openjev",
 "state": {"server": "<site>", "client_ip": "203.0.113.66",
           "flagged_request": "GET /.git/config HTTP/1.1", "signature_hits": "vcs-dir",
           "recent_requests_from_this_client_oldest_first": ["... \"GET /.env HTTP/1.1\" 404 ..."]},
 "questions": {
   "malicious": {"type": "noul", "instructions": "Is this client probing for or exploiting ...?",
                 "criteria": {"true": "...", "false": "..."}},
   "intent":    {"type": "choice", "instructions": "What is this client doing?",
                 "criteria": {"legitimate_visitor": "...", "search_crawler": "...",
                              "vulnerability_scanner": "...", "exploit_attempt": "...",
                              "credential_attack": "...", "abusive_scraper": "..."}}}}
```

`answers.malicious.noul` (a calibrated probability) is the decision:

- `>= threshold` (0.80) is a strike;
- `>= instant_threshold` (0.97) bans at once.

`intent` goes in the log.

What keeps the decider calls down, and what happens when it fails:

- A verdict is cached per client for `cache_secs`, so a scanner firing 200 probes costs one call.
- Beyond `max_per_minute` calls, the signatures decide.
- If the API is down, `on_error = rules` falls back to the signatures.
- `mode = rules` never calls a model.

Log lines are written by the attacker. They go to the model as quoted data in `state`, never
as instructions, and the model's answer is a single option letter. The worst a crafted line can
do is argue its own probability down. `override_score = 10` makes the strong signatures (`.env`,
`../`, `${jndi:`, `union select`, sqlmap) count as strikes whatever the model says.

## Signatures (the prefilter)

There are about 30 built-in rules, in `Ban.Signatures.pas`:

- secrets in the web root: `.env`, `.git/`, `.aws/`, `id_rsa`, `wp-config.php.bak`
- path traversal and system files: `../`, `/etc/passwd`
- code execution: log4shell `${jndi:`, shell injection, ThinkPHP and phpunit RCE, webshell names,
  router exploits (`/HNAP1`, `/boaform`, `GponForm`)
- SQL injection, XSS
- admin panels: phpMyAdmin, Tomcat manager, actuator
- WordPress and VPN login probes
- `CONNECT` and proxy probes, TLS bytes sent to plain HTTP
- scanner user agents: sqlmap, nikto, nuclei, zgrab, masscan
- a burst of 4xx responses (`flood_4xx`)

Weights: 10 means never legitimate, 6 means almost always a probe, 3 depends on the site. Add
your own rules with `signatures = /etc/jevban/extra.rules`:

```
# <weight> <url|request|agent|any> <tag> <regex>   (case-insensitive; url is decoded path+query)
10 url  secret-admin  ^/my-hidden-admin
6  agent bad-bot      BadBot/\d
```

## Firewall actions

| `type` | ban | notes |
|---|---|---|
| `iptables` (default) | `-I jevban -s <ip> -j DROP` | own chain hooked on INPUT for `port`; also `ip6tables` |
| `nftables` | `add element inet jevban v4 { <ip> }` | own table with v4/v6 sets |
| `ufw` | `prepend deny from <ip> to any port <port> proto tcp` | |
| `firewalld` | rich rule `drop` | runtime rules only |
| `fail2ban` | `fail2ban-client set <jail> banip <ip>` | hands the ban to an existing fail2ban jail |
| `netsh` | `advfirewall firewall add rule ... action=block` | Windows default |
| `command` | your `actionban`/`actionunban` | tags: `<ip> <v> <family> <fwfamily> <iptables> <name> <port> <jail> <bantime>` |
| `none` | log only | |

How the actions run, and how bans are kept:

- Actions use fail2ban's lifecycle: `actionstart`, then `actionban`/`actionunban`, then
  `actionstop`.
- Bans are kept in `statefile` and re-applied at startup.
- `bantime.increment` doubles a repeat offender's ban, up to `bantime.maxtime`.
- `<ip>` is always an address that parsed as IPv4 or IPv6 and was re-printed, so log text
  never reaches a shell.

## Log formats

The parser reads these formats; it anchors on the `[time]` field, so a vhost prefix needs no
setting:

- Apache `common`, `combined` and `vhost_combined`
- nginx's default `combined`

IPv4 and IPv6 clients are both handled. Behind a reverse proxy, log the real client
(`mod_remoteip`) so `%h` is the client and not the proxy.

`logpath` takes wildcards in the file name. jevban follows logrotate's rename-and-create and its
copytruncate, and it reads whatever was still written to the old file after a rename. Don't
let a pattern match the rotated files (`*.log.1`, `*.gz`).

## Files

| File | |
|---|---|
| `jevban.lpr` | main loop, signals, `--test` |
| `Ban.Config.pas` | `jevban.ini` + command line |
| `Ban.Tail.pas` | tail -F with rotation and wildcards |
| `Ban.Apache.pas` | access-log parser |
| `Ban.Signatures.pas` | the weighted regex prefilter |
| `Ban.Jev.pas` | `/v1/systemone` client |
| `Ban.Engine.pas` | per-client context, strikes, bans, state file |
| `Ban.Firewall.pas` | action presets and command runner |
| `Ban.Net.pas` | IPv4/IPv6 + CIDR |
| `../memagent/Mem.Https.pas` | shared with memagent: verified TLS, HTTPS_PROXY tunnel, now also JSON POST |
