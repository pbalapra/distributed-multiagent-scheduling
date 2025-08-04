from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from enum import Enum
import json
import uuid
from datetime import datetime

class MessageType(Enum):
    # Job management messages
    JOB_SUBMIT = "job_submit"
    JOB_SCHEDULE = "job_schedule"
    JOB_UPDATE = "job_update"
    JOB_COMPLETE = "job_complete"
    JOB_CANCEL = "job_cancel"
    
    # Resource management messages
    RESOURCE_ANNOUNCE = "resource_announce"
    RESOURCE_UPDATE = "resource_update"
    RESOURCE_REQUEST = "resource_request"
    RESOURCE_OFFER = "resource_offer"
    RESOURCE_RESERVATION = "resource_reservation"
    
    # Coordination messages
    HEARTBEAT = "heartbeat"
    NEGOTIATE = "negotiate"
    CONSENSUS = "consensus"
    LEADER_ELECTION = "leader_election"
    
    # System messages
    SYSTEM_STATUS = "system_status"
    ERROR = "error"
    ACK = "acknowledgment"

class MessagePriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class Message:
    """Represents a message between agents"""
    message_id: str
    message_type: MessageType
    sender_id: str
    recipient_id: str  # Can be "broadcast" for multicast messages
    timestamp: datetime
    priority: MessagePriority
    payload: Dict[str, Any]
    correlation_id: Optional[str] = None  # For linking request/response pairs
    ttl_seconds: int = 300  # Time to live
    
    def __post_init__(self):
        if not self.message_id:
            self.message_id = str(uuid.uuid4())
    
    def to_json(self) -> str:
        """Serialize message to JSON"""
        return json.dumps({
            'message_id': self.message_id,
            'message_type': self.message_type.value,
            'sender_id': self.sender_id,
            'recipient_id': self.recipient_id,
            'timestamp': self.timestamp.isoformat(),
            'priority': self.priority.value,
            'payload': self.payload,
            'correlation_id': self.correlation_id,
            'ttl_seconds': self.ttl_seconds
        })
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Message':
        """Deserialize message from JSON"""
        data = json.loads(json_str)
        return cls(
            message_id=data['message_id'],
            message_type=MessageType(data['message_type']),
            sender_id=data['sender_id'],
            recipient_id=data['recipient_id'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            priority=MessagePriority(data['priority']),
            payload=data['payload'],
            correlation_id=data.get('correlation_id'),
            ttl_seconds=data.get('ttl_seconds', 300)
        )
    
    def is_expired(self) -> bool:
        """Check if message has expired"""
        age_seconds = (datetime.now() - self.timestamp).total_seconds()
        return age_seconds > self.ttl_seconds

class MessageBus:
    """Simple message bus for agent communication"""
    
    def __init__(self):
        self.subscribers: Dict[str, List[callable]] = {}
        self.message_history: List[Message] = []
        self.max_history = 1000
    
    def subscribe(self, agent_id: str, callback: callable):
        """Subscribe an agent to receive messages"""
        if agent_id not in self.subscribers:
            self.subscribers[agent_id] = []
        self.subscribers[agent_id].append(callback)
    
    def unsubscribe(self, agent_id: str, callback: callable):
        """Unsubscribe an agent from receiving messages"""
        if agent_id in self.subscribers:
            self.subscribers[agent_id].remove(callback)
    
    def publish(self, message: Message):
        """Publish a message to interested agents"""
        # Add to history
        self.message_history.append(message)
        if len(self.message_history) > self.max_history:
            self.message_history.pop(0)
        
        # Skip expired messages
        if message.is_expired():
            return
        
        # Deliver to specific recipient or broadcast
        if message.recipient_id == "broadcast":
            # Broadcast to all subscribers except sender
            for agent_id, callbacks in self.subscribers.items():
                if agent_id != message.sender_id:
                    for callback in callbacks:
                        try:
                            callback(message)
                        except Exception as e:
                            print(f"Error delivering message to {agent_id}: {e}")
        else:
            # Deliver to specific recipient
            if message.recipient_id in self.subscribers:
                for callback in self.subscribers[message.recipient_id]:
                    try:
                        callback(message)
                    except Exception as e:
                        print(f"Error delivering message to {message.recipient_id}: {e}")
    
    def get_message_history(self, agent_id: str = None, 
                           message_type: MessageType = None) -> List[Message]:
        """Get message history with optional filtering"""
        messages = self.message_history
        
        if agent_id:
            messages = [m for m in messages 
                       if m.sender_id == agent_id or m.recipient_id == agent_id]
        
        if message_type:
            messages = [m for m in messages if m.message_type == message_type]
        
        return messages

# Message factory functions for common message types

def create_job_submit_message(sender_id: str, job_data: Dict[str, Any]) -> Message:
    """Create a job submission message"""
    return Message(
        message_id="",
        message_type=MessageType.JOB_SUBMIT,
        sender_id=sender_id,
        recipient_id="broadcast",
        timestamp=datetime.now(),
        priority=MessagePriority.NORMAL,
        payload={"job": job_data}
    )

def create_resource_offer_message(sender_id: str, recipient_id: str, 
                                 resource_data: Dict[str, Any], 
                                 job_id: str) -> Message:
    """Create a resource offer message"""
    return Message(
        message_id="",
        message_type=MessageType.RESOURCE_OFFER,
        sender_id=sender_id,
        recipient_id=recipient_id,
        timestamp=datetime.now(),
        priority=MessagePriority.HIGH,
        payload={
            "resource": resource_data,
            "job_id": job_id,
            "offer_expires": (datetime.now().timestamp() + 30)  # 30 second offer
        }
    )

def create_heartbeat_message(sender_id: str, status_data: Dict[str, Any]) -> Message:
    """Create a heartbeat message"""
    return Message(
        message_id="",
        message_type=MessageType.HEARTBEAT,
        sender_id=sender_id,
        recipient_id="broadcast",
        timestamp=datetime.now(),
        priority=MessagePriority.LOW,
        payload=status_data,
        ttl_seconds=60
    )

def create_negotiate_message(sender_id: str, recipient_id: str, 
                           job_id: str, proposal: Dict[str, Any]) -> Message:
    """Create a negotiation message"""
    return Message(
        message_id="",
        message_type=MessageType.NEGOTIATE,
        sender_id=sender_id,
        recipient_id=recipient_id,
        timestamp=datetime.now(),
        priority=MessagePriority.HIGH,
        payload={
            "job_id": job_id,
            "proposal": proposal
        }
    )

def create_ack_message(sender_id: str, recipient_id: str, 
                      correlation_id: str, success: bool = True) -> Message:
    """Create an acknowledgment message"""
    return Message(
        message_id="",
        message_type=MessageType.ACK,
        sender_id=sender_id,
        recipient_id=recipient_id,
        timestamp=datetime.now(),
        priority=MessagePriority.NORMAL,
        payload={"success": success},
        correlation_id=correlation_id
    )
