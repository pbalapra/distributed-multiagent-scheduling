import threading
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging

from ..communication.protocol import Message, MessageBus, MessageType, MessagePriority

class BaseAgent(ABC):
    """Base class for all agents in the HPC scheduler system"""
    
    def __init__(self, agent_id: str, message_bus: MessageBus, 
                 heartbeat_interval: float = 30.0):
        self.agent_id = agent_id
        self.message_bus = message_bus
        self.heartbeat_interval = heartbeat_interval
        self.is_running = False
        self.last_heartbeat = datetime.now()
        
        # Threading
        self._main_thread = None
        self._heartbeat_thread = None
        self._stop_event = threading.Event()
        
        # Message handling
        self.message_handlers = {
            MessageType.HEARTBEAT: self._handle_heartbeat,
            MessageType.ACK: self._handle_ack,
            MessageType.ERROR: self._handle_error,
        }
        
        # Metrics
        self.messages_sent = 0
        self.messages_received = 0
        self.start_time = datetime.now()
        
        # Setup logging
        self.logger = logging.getLogger(f"Agent-{agent_id}")
        
        # Subscribe to message bus
        self.message_bus.subscribe(self.agent_id, self._on_message)
    
    def start(self):
        """Start the agent"""
        if self.is_running:
            return
            
        self.is_running = True
        self._stop_event.clear()
        
        # Start main agent thread
        self._main_thread = threading.Thread(target=self._run, daemon=True)
        self._main_thread.start()
        
        # Start heartbeat thread
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()
        
        self.logger.info(f"Agent {self.agent_id} started")
    
    def stop(self):
        """Stop the agent"""
        if not self.is_running:
            return
            
        self.is_running = False
        self._stop_event.set()
        
        # Wait for threads to finish
        if self._main_thread and self._main_thread.is_alive():
            self._main_thread.join(timeout=5.0)
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=5.0)
            
        self.logger.info(f"Agent {self.agent_id} stopped")
    
    def send_message(self, message: Message):
        """Send a message through the message bus"""
        self.message_bus.publish(message)
        self.messages_sent += 1
        self.logger.debug(f"Sent {message.message_type.value} to {message.recipient_id}")
    
    def _on_message(self, message: Message):
        """Handle incoming messages"""
        self.messages_received += 1
        
        # Check if we have a handler for this message type
        if message.message_type in self.message_handlers:
            try:
                self.message_handlers[message.message_type](message)
            except Exception as e:
                self.logger.error(f"Error handling {message.message_type.value}: {e}")
        else:
            self.logger.warning(f"No handler for message type: {message.message_type.value}")
    
    def _handle_heartbeat(self, message: Message):
        """Handle heartbeat messages from other agents"""
        if message.sender_id != self.agent_id:
            self.logger.debug(f"Received heartbeat from {message.sender_id}")
    
    def _handle_ack(self, message: Message):
        """Handle acknowledgment messages"""
        self.logger.debug(f"Received ACK from {message.sender_id}")
    
    def _handle_error(self, message: Message):
        """Handle error messages"""
        self.logger.error(f"Received error from {message.sender_id}: {message.payload}")
    
    def _heartbeat_loop(self):
        """Send periodic heartbeat messages"""
        while self.is_running and not self._stop_event.is_set():
            try:
                self._send_heartbeat()
                self._stop_event.wait(self.heartbeat_interval)
            except Exception as e:
                self.logger.error(f"Error in heartbeat loop: {e}")
    
    def _send_heartbeat(self):
        """Send a heartbeat message"""
        heartbeat_data = self._get_heartbeat_data()
        message = Message(
            message_id="",
            message_type=MessageType.HEARTBEAT,
            sender_id=self.agent_id,
            recipient_id="broadcast",
            timestamp=datetime.now(),
            priority=MessagePriority.LOW,
            payload=heartbeat_data,
            ttl_seconds=60
        )
        self.send_message(message)
        self.last_heartbeat = datetime.now()
    
    def _get_heartbeat_data(self) -> Dict[str, Any]:
        """Get data to include in heartbeat messages"""
        return {
            "agent_type": self.__class__.__name__,
            "status": "running",
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds(),
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received
        }
    
    @abstractmethod
    def _run(self):
        """Main agent loop - to be implemented by subclasses"""
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.__class__.__name__,
            "is_running": self.is_running,
            "start_time": self.start_time.isoformat(),
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "uptime_seconds": (datetime.now() - self.start_time).total_seconds()
        }
