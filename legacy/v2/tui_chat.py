#!/usr/bin/env python3

import asyncio
import logging
import threading
import sys
import datetime
from typing import Optional, List, Tuple

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Button, Footer, Header, Input, RichLog, Static, 
    ProgressBar, Label
)
from textual.reactive import reactive
from textual.message import Message
from textual.binding import Binding
from textual import log
from rich.text import Text
from rich.panel import Panel
from rich.markdown import Markdown

import cragchat
import run


class SystemStatus(Static):
    """Widget to display system status and controls."""
    
    def __init__(self) -> None:
        super().__init__()
        self.status = "Initializing..."
    
    def compose(self) -> ComposeResult:
        with Container(id="status-container"):
            yield Label("🤖 CRAG Chat System", id="title")
            yield Label(self.status, id="status-text")
            yield Button("🔄 Train LoRA", id="train-btn", variant="primary")
            yield Button("🗑️ Clear Embeddings", id="clear-btn", variant="warning")
            yield Button("💾 Save Session", id="save-btn", variant="success")


class ChatMessage(Static):
    """Widget to display a single chat message."""
    
    def __init__(self, sender: str, content: str, timestamp: str = None) -> None:
        super().__init__()
        self.sender = sender
        self.content = content
        self.timestamp = timestamp or datetime.datetime.now().strftime("%H:%M:%S")
    
    def compose(self) -> ComposeResult:
        if self.sender == "user":
            yield Static(f"[bold blue]You[/] ({self.timestamp}): {self.content}", 
                        classes="user-message")
        elif self.sender == "assistant":
            # Parse analysis and answer tags
            analysis_match = cragchat.re.search(r'<analysis>(.*?)</analysis>', self.content, cragchat.re.DOTALL)
            answer_match = cragchat.re.search(r'<answer>(.*?)</answer>', self.content, cragchat.re.DOTALL)
            
            if analysis_match and answer_match:
                analysis = analysis_match.group(1).strip()
                answer = answer_match.group(1).strip()
                
                yield Static(f"[bold green]Assistant[/] ({self.timestamp}):", 
                            classes="assistant-header")
                yield Static(Panel(analysis, title="Analysis", border_style="dim"), 
                            classes="analysis-panel")
                yield Static(Panel(answer, title="Answer", border_style="green"), 
                            classes="answer-panel")
            else:
                yield Static(f"[bold green]Assistant[/] ({self.timestamp}): {self.content}", 
                            classes="assistant-message")
        else:
            yield Static(f"[bold yellow]{self.sender}[/] ({self.timestamp}): {self.content}", 
                        classes="system-message")


class ChatLog(ScrollableContainer):
    """Container for chat messages with auto-scroll."""
    
    def __init__(self) -> None:
        super().__init__(id="chat-log")
        self.messages: List[ChatMessage] = []
    
    def add_message(self, sender: str, content: str) -> None:
        """Add a new message to the chat log."""
        message = ChatMessage(sender, content)
        self.messages.append(message)
        self.mount(message)
        # Auto-scroll to bottom
        self.call_after_refresh(self.scroll_end)
    
    def clear_messages(self) -> None:
        """Clear all messages from the chat log."""
        for message in self.messages:
            message.remove()
        self.messages.clear()


class CragChatApp(App):
    """Main TUI application for CRAG Chat."""
    
    CSS = """
    Screen {
        layout: grid;
        grid-size: 4 4;
        grid-gutter: 1;
    }
    
    #sidebar {
        column-span: 1;
        row-span: 4;
        background: $surface;
        border: solid $primary;
    }
    
    #chat-area {
        column-span: 3;
        row-span: 3;
    }
    
    #input-area {
        column-span: 3;
        row-span: 1;
        background: $surface;
        border: solid $accent;
    }
    
    #chat-log {
        height: 100%;
        background: $background;
        border: solid $secondary;
    }
    
    #message-input {
        width: 100%;
    }
    
    .user-message {
        color: $text;
        background: $primary 20%;
        margin: 1;
        padding: 1;
    }
    
    .assistant-message {
        color: $text;
        background: $success 20%;
        margin: 1;
        padding: 1;
    }
    
    .assistant-header {
        color: $success;
        margin-top: 1;
        margin-bottom: 1;
    }
    
    .analysis-panel {
        margin-bottom: 1;
    }
    
    .answer-panel {
        margin-bottom: 1;
    }
    
    .system-message {
        color: $warning;
        background: $warning 20%;
        margin: 1;
        padding: 1;
    }
    
    #status-container {
        padding: 1;
        height: 100%;
    }
    
    #title {
        text-align: center;
        margin-bottom: 1;
        color: $primary;
    }
    
    #status-text {
        margin-bottom: 2;
        text-align: center;
    }
    
    Button {
        width: 100%;
        margin-bottom: 1;
    }
    
    #progress-container {
        height: 3;
        margin: 1;
    }
    """
    
    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
        Binding("ctrl+t", "train_lora", "Train LoRA"),
        Binding("ctrl+e", "clear_embeddings", "Clear Embeddings"),
        Binding("ctrl+s", "save_session", "Save Session"),
    ]
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.tokenizer = None
        self.collection = None
        self.embedder = None
        self.logger = None
        self.is_processing = False
        self.log_file_name = f"cragchat_tui_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    def compose(self) -> ComposeResult:
        """Create the layout."""
        yield Header()
        
        with Container(id="sidebar"):
            yield SystemStatus()
        
        with Container(id="chat-area"):
            yield ChatLog()
        
        with Container(id="input-area"):
            yield Input(placeholder="Ask a question or type a command...", id="message-input")
        
        yield Footer()
    
    async def on_mount(self) -> None:
        """Initialize the system when app starts."""
        self.chat_log = self.query_one(ChatLog)
        self.message_input = self.query_one("#message-input", Input)
        self.status_widget = self.query_one(SystemStatus)
        
        # Set focus on input
        self.message_input.focus()
        
        # Initialize system in background
        await self.initialize_system()
    
    async def initialize_system(self) -> None:
        """Initialize the CRAG chat system."""
        self.status_widget.status = "Initializing system..."
        self.status_widget.query_one("#status-text", Label).update("Initializing system...")
        
        try:
            # Setup logging
            self.logger = run.setup_logging(self.log_file_name)
            self.logger.info("=== CRAGCHAT TUI SESSION STARTED ===")
            self.logger.info(f"Log file: {self.log_file_name}")
            
            # Ensure chat history exists
            run.ensure_chat_history_exists()
            
            # Initialize in a separate thread to avoid blocking
            def init_models():
                try:
                    self.model, self.tokenizer, self.collection, self.embedder = cragchat.main_jupyter()
                    return True
                except Exception as e:
                    self.logger.error(f"Initialization error: {e}")
                    return False
            
            # Run initialization in thread
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(None, init_models)
            
            if success and self.model is not None:
                self.status_widget.status = "✅ Ready"
                self.status_widget.query_one("#status-text", Label).update("✅ Ready")
                self.chat_log.add_message("system", "🤖 CRAG Chat System initialized and ready!")
            else:
                self.status_widget.status = "❌ Failed"
                self.status_widget.query_one("#status-text", Label).update("❌ Initialization failed")
                self.chat_log.add_message("system", "❌ Failed to initialize system. Check logs for details.")
                
        except Exception as e:
            self.logger.error(f"Initialization error: {e}")
            self.status_widget.status = "❌ Error"
            self.status_widget.query_one("#status-text", Label).update("❌ Error")
            self.chat_log.add_message("system", f"❌ Error during initialization: {str(e)}")
    
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle user input submission."""
        if self.is_processing:
            return
            
        user_input = event.value.strip()
        if not user_input:
            return
            
        # Clear input
        self.message_input.value = ""
        
        # Add user message to chat
        self.chat_log.add_message("user", user_input)
        
        # Handle commands and questions
        await self.process_user_input(user_input)
    
    async def process_user_input(self, user_input: str) -> None:
        """Process user input (commands or questions)."""
        self.is_processing = True
        
        try:
            if user_input.lower() == '<train_lora>':
                await self.train_lora()
            elif user_input.lower() == '<clear_embeddings>':
                await self.clear_embeddings()
            elif user_input.lower() == 'exit':
                await self.action_quit()
            else:
                await self.ask_question(user_input)
        finally:
            self.is_processing = False
    
    async def ask_question(self, question: str) -> None:
        """Ask a question to the model."""
        if not self.model:
            self.chat_log.add_message("system", "❌ System not initialized. Please wait for initialization to complete.")
            return
            
        self.status_widget.query_one("#status-text", Label).update("🤔 Thinking...")
        
        try:
            # Run inference in a separate thread
            def get_response():
                return cragchat.chat_and_record_jupyter(
                    self.model, self.tokenizer, self.collection, self.embedder, question
                )
            
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(None, get_response)
            
            # Log interaction
            if self.logger:
                run.log_interaction(self.logger, question, response)
            
            # Add response to chat
            self.chat_log.add_message("assistant", response)
            
        except Exception as e:
            error_msg = f"❌ Error processing question: {str(e)}"
            self.chat_log.add_message("system", error_msg)
            if self.logger:
                self.logger.error(f"Question processing error: {e}")
        finally:
            self.status_widget.query_one("#status-text", Label).update("✅ Ready")
    
    async def train_lora(self) -> None:
        """Train LoRA adapter."""
        self.chat_log.add_message("system", "🔄 Starting LoRA training...")
        self.status_widget.query_one("#status-text", Label).update("🔄 Training...")
        
        try:
            # Run training in a separate thread
            def run_training():
                return cragchat.train_lora()
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, run_training)
            
            self.chat_log.add_message("system", "✅ LoRA training completed!")
            
            # Reload model
            self.chat_log.add_message("system", "🔄 Reloading model with new adapter...")
            
            def reload_models():
                return cragchat.main_jupyter()
            
            self.model, self.tokenizer, self.collection, self.embedder = await loop.run_in_executor(None, reload_models)
            
            if self.model:
                self.chat_log.add_message("system", "✅ Model reloaded successfully!")
                if self.logger:
                    self.logger.info("LoRA training and model reload completed successfully")
            else:
                self.chat_log.add_message("system", "❌ Failed to reload model after training")
                
        except Exception as e:
            error_msg = f"❌ Training failed: {str(e)}"
            self.chat_log.add_message("system", error_msg)
            if self.logger:
                self.logger.error(f"Training error: {e}")
        finally:
            self.status_widget.query_one("#status-text", Label).update("✅ Ready")
    
    async def clear_embeddings(self) -> None:
        """Clear and regenerate embeddings."""
        if not self.collection or not self.tokenizer:
            self.chat_log.add_message("system", "❌ System not initialized")
            return
            
        self.chat_log.add_message("system", "🗑️ Clearing embeddings...")
        self.status_widget.query_one("#status-text", Label).update("🗑️ Clearing...")
        
        try:
            # Run in separate thread
            def clear_embeddings():
                return cragchat.clear_and_regenerate_embeddings(self.collection, self.tokenizer)
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, clear_embeddings)
            
            self.chat_log.add_message("system", "✅ Embeddings cleared and regenerated!")
            if self.logger:
                self.logger.info("Embeddings cleared and regenerated successfully")
                
        except Exception as e:
            error_msg = f"❌ Failed to clear embeddings: {str(e)}"
            self.chat_log.add_message("system", error_msg)
            if self.logger:
                self.logger.error(f"Embedding clearance error: {e}")
        finally:
            self.status_widget.query_one("#status-text", Label).update("✅ Ready")
    
    async def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "train-btn":
            await self.train_lora()
        elif event.button.id == "clear-btn":
            await self.clear_embeddings()
        elif event.button.id == "save-btn":
            self.chat_log.add_message("system", f"💾 Session saved to: {self.log_file_name}")
    
    def action_train_lora(self) -> None:
        """Action to train LoRA (keyboard shortcut)."""
        asyncio.create_task(self.train_lora())
    
    def action_clear_embeddings(self) -> None:
        """Action to clear embeddings (keyboard shortcut)."""
        asyncio.create_task(self.clear_embeddings())
    
    def action_save_session(self) -> None:
        """Action to save session (keyboard shortcut)."""
        self.chat_log.add_message("system", f"💾 Session saved to: {self.log_file_name}")


def main():
    """Run the TUI chat application."""
    if len(sys.argv) > 1:
        print("TUI Chat App - Custom log filename not supported in TUI mode")
        print("Logs will be automatically saved with timestamp")
    
    app = CragChatApp()
    app.run()


if __name__ == "__main__":
    main()