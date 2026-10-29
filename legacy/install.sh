curl -fsSL https://ollama.com/install.sh | sh
ollama serve &

mkdir -p examples/data/config
curl -L \
  https://raw.githubusercontent.com/MemTensor/MemOS/main/examples/data/config/simple_memos_config.json \
  -o examples/data/config/simple_memos_config.json

git clone https://github.com/MemTensor/MemOS.git MemOS-repo
mkdir -p examples/data
cp -r MemOS-repo/examples/data/mem_cube_2 examples/data/mem_cube_2