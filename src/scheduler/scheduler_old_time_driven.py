import threading
import time
from datetime import datetime
from typing import Dict, Any, List

from ..communication.protocol import MessageBus, MessageType, MessagePriority, create_ack_message, Message
from ..jobs.job import Job, JobStatus
from .job_pool import JobPool

class Scheduler:
    """The central scheduler coordinating jobs across resource agents"""

    def __init__(self, message_bus: MessageBus, job_pool: JobPool):
        self.message_bus = message_bus
        self.job_pool = job_pool
        self.message_bus.subscribe('scheduler', self._on_message)
        
        self.pending_offers: Dict[str, Dict] = {}
        self.negotiations: Dict[str, Dict] = {}
        
        self._stop_event = threading.Event()
        self._main_thread = None
        
    def start(self):
        """Start the scheduler"""
        self._main_thread = threading.Thread(target=self._run, daemon=True)
        self._main_thread.start()
    
    def stop(self):
        """Stop the scheduler"""
        self._stop_event.set()
        if self._main_thread and self._main_thread.is_alive():
            self._main_thread.join()

    def _run(self):
        """Main loop for the scheduler"""
        while not self._stop_event.is_set():
            try:
                # Implement scheduling logic here

                # Placeholder: just log a simple status
                print(f"Scheduler running at {datetime.now()}")
                
                # Sleep before next iteration
                self._stop_event.wait(10)
            except Exception as e:
                print(f"Scheduler error: {e}")

    def _on_message(self, message):
        """Handle incoming messages to the scheduler"""
        if message.message_type == MessageType.JOB_COMPLETE:
            self._handle_job_complete(message)
        elif message.message_type == MessageType.RESOURCE_OFFER:
            self._handle_resource_offer(message)
        elif message.message_type == MessageType.NEGOTIATE:
            self._handle_negotiate(message)
        # Add more handlers as necessary
        
    def _handle_job_complete(self, message):
        """Handle job completion messages"""
        job_id = message.payload.get('job_id')
        status = message.payload.get('status')
        if job_id and status:
            # Convert string status to JobStatus enum
            try:
                job_status = JobStatus(status)
                self.job_pool.update_job_status(job_id, job_status)
                print(f"✅ Job {job_id} completed with status: {status}")
            except ValueError:
                print(f"❌ Invalid job status: {status} for job {job_id}")

    def _handle_resource_offer(self, message):
        """Handle resource offers from resource agents"""
        offer = message.payload.get('resource')
        job_id = message.payload.get('job_id')
        sender_id = message.sender_id
        
        if job_id not in self.pending_offers:
            self.pending_offers[job_id] = []
        
        # Add sender info to offer
        offer['agent_id'] = sender_id
        self.pending_offers[job_id].append(offer)
        
        print(f"📨 Received offer for job {job_id} from {sender_id}, score: {offer.get('score', 'N/A')}")
        
        # Evaluate offers and make decisions immediately
        self._evaluate_offers(job_id)

    def _handle_negotiate(self, message):
        """Handle negotiation messages"""
        job_id = message.payload.get('job_id')
        response = message.payload.get('response')
        proposal = message.payload.get('counter_proposal')
        
        if job_id:
            print(f"Negotiation for job {job_id}: {response}")
            self.negotiations[job_id] = {
                'response': response,
                'proposal': proposal
            }

    def _evaluate_offers(self, job_id):
        """Evaluate resource offers for a given job"""
        if job_id not in self.pending_offers or not self.pending_offers[job_id]:
            return
            
        offers = self.pending_offers[job_id]
        
        # Simple strategy: pick the offer with the highest score
        best_offer = max(offers, key=lambda x: x.get('score', 0))
        best_agent = best_offer.get('agent_id')
        
        if best_agent:
            print(f"🎯 Assigning job {job_id} to {best_agent} (score: {best_offer.get('score', 'N/A')})")
            
            # Get the job from job pool
            job = self.job_pool.get_job(job_id)
            if job:
                # Send resource reservation to the selected agent
                reservation_message = Message(
                    message_id="",
                    message_type=MessageType.RESOURCE_RESERVATION,
                    sender_id="scheduler",
                    recipient_id=best_agent,
                    timestamp=datetime.now(),
                    priority=MessagePriority.HIGH,
                    payload={"job": job.to_dict()}
                )
                self.message_bus.publish(reservation_message)
                
                # Clear pending offers for this job
                del self.pending_offers[job_id]
