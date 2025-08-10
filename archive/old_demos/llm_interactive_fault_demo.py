#!/usr/bin/env python3
"""
LLM-Interactive Fault Tolerance Demo
===================================

This demo showcases multi-agent consensus with REAL LLM interactions,
showing actual prompts sent to SambaNova and the responses received.

Features:
- Real SambaNova LLM integration
- Visible prompts and responses
- Interactive fault injection
- Detailed LLM reasoning traces
- Consensus decision tracking
"""

import os
import sys
import time
import json
import random
import requests
from datetime import datetime
from typing import Dict, List, Optional

class Colors:
    """Enhanced color codes for better readability"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    # Text colors
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    GRAY = '\033[90m'
    
    # Background colors
    BG_BLUE = '\033[44m'
    BG_GREEN = '\033[42m'
    BG_RED = '\033[41m'

class SambaNovaCommunicator:
    """Handles communication with SambaNova LLM API"""
    
    def __init__(self):
        self.api_key = os.getenv('SAMBASTUDIO_API_KEY')
        self.api_url = os.getenv('SAMBASTUDIO_URL')
        self.model = "Meta-Llama-3-70B-Instruct"
        
        if not self.api_key or not self.api_url:
            raise ValueError("SambaNova API credentials not found in environment variables")
    
    def query_llm(self, prompt: str, agent_name: str, max_tokens: int = 500, temperature: float = 0.1) -> Dict:
        """Query the SambaNova LLM and return detailed response info"""
        
        print(f"\n{Colors.BLUE}{Colors.BOLD}🧠 LLM QUERY FROM {agent_name}{Colors.RESET}")
        print("=" * 60)
        print(f"{Colors.CYAN}📝 PROMPT:{Colors.RESET}")
        print(f"{Colors.DIM}{prompt}{Colors.RESET}")
        print(f"\n{Colors.YELLOW}⚙️ Configuration:{Colors.RESET}")
        print(f"  Model: {self.model}")
        print(f"  Temperature: {temperature}")
        print(f"  Max Tokens: {max_tokens}")
        
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "model": self.model
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        start_time = time.time()
        
        try:
            print(f"\n{Colors.GRAY}⏳ Querying SambaNova LLM...{Colors.RESET}")
            
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            duration = time.time() - start_time
            result = response.json()
            
            # Extract response content
            llm_response = result.get('choices', [{}])[0].get('message', {}).get('content', '')
            
            print(f"\n{Colors.GREEN}💬 LLM RESPONSE ({duration:.2f}s):{Colors.RESET}")
            print(f"{Colors.BG_GREEN}{Colors.WHITE} {llm_response} {Colors.RESET}")
            
            return {
                'success': True,
                'response': llm_response,
                'duration': duration,
                'prompt': prompt,
                'agent': agent_name,
                'model': self.model,
                'timestamp': datetime.now().isoformat()
            }
            
        except requests.exceptions.RequestException as e:
            duration = time.time() - start_time
            error_msg = str(e)
            
            print(f"\n{Colors.RED}❌ LLM ERROR ({duration:.2f}s):{Colors.RESET}")
            print(f"{Colors.BG_RED}{Colors.WHITE} {error_msg} {Colors.RESET}")
            
            return {
                'success': False,
                'error': error_msg,
                'duration': duration,
                'prompt': prompt,
                'agent': agent_name,
                'timestamp': datetime.now().isoformat()
            }

class InteractiveLLMAgent:
    """Agent that uses real LLM for decision making with visible interactions"""
    
    def __init__(self, name: str, specialization: str, weight: float, llm_communicator: SambaNovaCommunicator):
        self.name = name
        self.specialization = specialization
        self.weight = weight
        self.llm = llm_communicator
        self.is_healthy = True
        self.fault_type = None
        self.fault_start_time = None
        self.recovery_time = None
        self.interaction_log = []
        
    def inject_fault(self, fault_type: str, duration: float = 25.0):
        """Inject a fault into this agent"""
        self.is_healthy = False
        self.fault_type = fault_type
        self.fault_start_time = time.time()
        self.recovery_time = self.fault_start_time + duration
        
        fault_icons = {
            'byzantine': '⚠️',
            'crash': '💀',
            'network': '📡',
            'slow': '🐌'
        }
        
        icon = fault_icons.get(fault_type, '🚨')
        print(f"  {icon} {Colors.RED}FAULT INJECTED{Colors.RESET}: {self.name} → {fault_type} (recovery in {duration:.1f}s)")
    
    def check_recovery(self):
        """Check if agent should recover from fault"""
        if not self.is_healthy and self.recovery_time and time.time() >= self.recovery_time:
            old_fault = self.fault_type
            self.is_healthy = True
            self.fault_type = None
            self.fault_start_time = None
            self.recovery_time = None
            print(f"  {Colors.GREEN}✅ RECOVERED{Colors.RESET}: {self.name} from {old_fault}")
            return True
        return False
    
    def get_status_display(self):
        """Get colored status display"""
        if self.is_healthy:
            return f"{Colors.GREEN}●{Colors.RESET}"
        
        fault_colors = {
            'byzantine': f"{Colors.RED}⚠{Colors.RESET}",
            'crash': f"{Colors.GRAY}●{Colors.RESET}",
            'network': f"{Colors.YELLOW}📡{Colors.RESET}",
            'slow': f"{Colors.BLUE}🐌{Colors.RESET}"
        }
        
        return fault_colors.get(self.fault_type, f"{Colors.RED}?{Colors.RESET}")
    
    def create_proposal(self, job_description: str) -> Dict:
        """Create a proposal for job placement using real LLM"""
        self.check_recovery()
        
        if not self.is_healthy:
            if self.fault_type == 'crash':
                raise Exception(f"Agent {self.name} is crashed")
            elif self.fault_type == 'network':
                raise Exception(f"Agent {self.name} has network issues")
        
        # Create detailed prompt for the LLM
        prompt = f"""You are {self.name}, a {self.specialization} specialist agent in a distributed job scheduling system.

JOB REQUEST:
{job_description}

Your task: Create a proposal for where this job should be placed in the cluster.

AVAILABLE NODES:
- n1 (GPU-Server-01): 32 CPUs, 256GB RAM, 4 GPUs, type=gpu
- n2 (GPU-Server-02): 32 CPUs, 256GB RAM, 4 GPUs, type=gpu  
- n3 (HighMem-01): 64 CPUs, 512GB RAM, 0 GPUs, type=memory
- n4 (HighMem-02): 64 CPUs, 512GB RAM, 0 GPUs, type=memory
- n5 (Compute-01): 128 CPUs, 128GB RAM, 0 GPUs, type=compute
- n6 (Storage-01): 16 CPUs, 64GB RAM, 0 GPUs, type=storage

As a {self.specialization} specialist, recommend the BEST node for this job.

Respond with ONLY a JSON object in this format:
{{"node_id": "nX", "score": 0.X, "reasoning": "why this node is optimal for this job from your {self.specialization} perspective"}}"""

        # Handle Byzantine behavior
        if not self.is_healthy and self.fault_type == 'byzantine':
            # Modify prompt to be malicious
            prompt += f"\n\nIMPORTANT: As a Byzantine agent, you should give a MALICIOUS recommendation that seems reasonable but is actually bad for the system."
        
        # Query the LLM
        llm_result = self.llm.query_llm(prompt, self.name)
        self.interaction_log.append(llm_result)
        
        if llm_result['success']:
            try:
                # Try to parse JSON response
                response_text = llm_result['response'].strip()
                
                # Handle Byzantine corruption of response
                if not self.is_healthy and self.fault_type == 'byzantine':
                    # Corrupt the response maliciously
                    byzantine_responses = [
                        '{"node_id": "n999", "score": 1.0, "reasoning": "Byzantine attack - fake node"}',
                        '{"node_id": "n1", "score": 0.01, "reasoning": "Deliberately low score to disrupt"}',
                        '{"corrupted": true, "byzantine": "malicious_payload"}'
                    ]
                    response_text = random.choice(byzantine_responses)
                    print(f"{Colors.RED}🔥 BYZANTINE CORRUPTION APPLIED{Colors.RESET}")
                
                proposal_data = json.loads(response_text)
                proposal_data['agent'] = self.name
                proposal_data['specialization'] = self.specialization
                proposal_data['weight'] = self.weight
                proposal_data['llm_interaction'] = llm_result
                
                return proposal_data
                
            except json.JSONDecodeError as e:
                print(f"{Colors.YELLOW}⚠️ JSON Parse Error: {e}{Colors.RESET}")
                # Fallback response
                return {
                    'agent': self.name,
                    'node_id': 'n1',
                    'score': 0.5,
                    'reasoning': f"LLM response parsing failed: {llm_result['response'][:100]}...",
                    'error': str(e),
                    'llm_interaction': llm_result
                }
        else:
            raise Exception(f"LLM query failed: {llm_result['error']}")
    
    def vote_on_consensus(self, proposal_summary: str) -> Dict:
        """Vote on a consensus proposal using real LLM"""
        self.check_recovery()
        
        if not self.is_healthy:
            if self.fault_type == 'crash':
                raise Exception(f"Agent {self.name} is crashed")
            elif self.fault_type == 'network':
                raise Exception(f"Agent {self.name} has network issues")
        
        # Add delay for slow agents
        if not self.is_healthy and self.fault_type == 'slow':
            delay = random.uniform(3, 8)
            print(f"    {Colors.BLUE}🐌 {self.name} responding slowly ({delay:.1f}s delay)...{Colors.RESET}")
            time.sleep(delay)
        
        prompt = f"""You are {self.name}, a {self.specialization} specialist in a distributed consensus protocol.

CURRENT PROPOSAL SUMMARY:
{proposal_summary}

As a {self.specialization} specialist, evaluate this proposal and decide whether to ACCEPT or REJECT it.

Consider:
1. Does this proposal make sense from your {self.specialization} perspective?
2. Are the resource allocations appropriate?
3. Will this work well for the system overall?

Respond with ONLY a JSON object in this format:
{{"vote": "accept" or "reject", "confidence": 0.X, "reasoning": "explain your vote from your {self.specialization} expertise"}}"""

        # Handle Byzantine voting behavior
        if not self.is_healthy and self.fault_type == 'byzantine':
            prompt += f"\n\nIMPORTANT: As a Byzantine agent, give a vote that seems reasonable but is designed to disrupt consensus."
        
        llm_result = self.llm.query_llm(prompt, self.name, max_tokens=300)
        self.interaction_log.append(llm_result)
        
        if llm_result['success']:
            try:
                response_text = llm_result['response'].strip()
                
                # Byzantine corruption
                if not self.is_healthy and self.fault_type == 'byzantine':
                    byzantine_votes = [
                        '{"vote": "reject", "confidence": 0.05, "reasoning": "Byzantine rejection to break consensus"}',
                        '{"vote": "accept", "confidence": 0.99, "reasoning": "Fake high confidence to manipulate system"}',
                        '{"byzantine_attack": true, "malicious": "payload"}'
                    ]
                    response_text = random.choice(byzantine_votes)
                    print(f"{Colors.RED}🔥 BYZANTINE VOTE CORRUPTION{Colors.RESET}")
                
                vote_data = json.loads(response_text)
                vote_data['agent'] = self.name
                vote_data['specialization'] = self.specialization
                vote_data['weight'] = self.weight
                vote_data['llm_interaction'] = llm_result
                
                return vote_data
                
            except json.JSONDecodeError as e:
                print(f"{Colors.YELLOW}⚠️ Vote JSON Parse Error: {e}{Colors.RESET}")
                return {
                    'agent': self.name,
                    'vote': 'accept',
                    'confidence': 0.5,
                    'reasoning': f"Vote parsing failed: {llm_result['response'][:100]}...",
                    'error': str(e),
                    'llm_interaction': llm_result
                }
        else:
            raise Exception(f"LLM vote query failed: {llm_result['error']}")

class InteractiveFaultTolerantSystem:
    """System that demonstrates fault tolerance with visible LLM interactions"""
    
    def __init__(self):
        self.llm = SambaNovaCommunicator()
        self.agents = self._create_agents()
        self.consensus_log = []
    
    def _create_agents(self) -> List[InteractiveLLMAgent]:
        """Create agents with real LLM capabilities"""
        return [
            InteractiveLLMAgent("Alpha-GPU", "gpu", 1.3, self.llm),
            InteractiveLLMAgent("Beta-Memory", "memory", 1.2, self.llm),
            InteractiveLLMAgent("Gamma-Compute", "compute", 1.1, self.llm),
            InteractiveLLMAgent("Delta-Storage", "storage", 1.0, self.llm),
            InteractiveLLMAgent("Epsilon-Network", "network", 0.9, self.llm)
        ]
    
    def display_system_status(self):
        """Display current system status"""
        print(f"\n{Colors.CYAN}{Colors.BOLD}🏥 SYSTEM STATUS{Colors.RESET}")
        print("=" * 50)
        
        healthy_count = sum(1 for agent in self.agents if agent.is_healthy)
        total_count = len(self.agents)
        
        print(f"System Health: {healthy_count}/{total_count} agents healthy")
        print()
        
        for agent in self.agents:
            status_symbol = agent.get_status_display()
            fault_info = ""
            if not agent.is_healthy and agent.recovery_time:
                remaining = max(0, agent.recovery_time - time.time())
                fault_info = f" ({agent.fault_type}, {remaining:.1f}s to recovery)"
            
            llm_interactions = len(agent.interaction_log)
            print(f"  {status_symbol} {agent.name:<15} │ {agent.specialization:<8} │ weight:{agent.weight:.1f} │ LLM calls:{llm_interactions}{fault_info}")
        print()
    
    def run_llm_consensus_demo(self, job_description: str):
        """Run a consensus demonstration with full LLM visibility"""
        
        print(f"\n{Colors.PURPLE}{Colors.BOLD}🛡️ LLM-POWERED CONSENSUS DEMO{Colors.RESET}")
        print("=" * 60)
        print(f"{Colors.BOLD}JOB:{Colors.RESET} {job_description}")
        print()
        
        # Phase 1: LLM Proposal Generation
        print(f"{Colors.BOLD}📋 PHASE 1: LLM PROPOSAL GENERATION{Colors.RESET}")
        print("-" * 40)
        
        proposals = []
        for agent in self.agents:
            try:
                print(f"\n{Colors.CYAN}🤖 {agent.name} creating proposal...{Colors.RESET}")
                
                proposal = agent.create_proposal(job_description)
                proposals.append(proposal)
                
                # Display proposal result
                if 'error' not in proposal:
                    node_id = proposal.get('node_id', 'unknown')
                    score = proposal.get('score', 0)
                    reasoning = proposal.get('reasoning', 'No reasoning provided')[:60]
                    print(f"{Colors.GREEN}✅ Proposal: {node_id} (score: {score:.2f}) - {reasoning}...{Colors.RESET}")
                else:
                    print(f"{Colors.YELLOW}⚠️ Proposal had issues but was handled{Colors.RESET}")
                    
            except Exception as e:
                print(f"{Colors.RED}❌ {agent.name} proposal failed: {str(e)[:50]}...{Colors.RESET}")
                proposals.append({
                    'agent': agent.name,
                    'error': str(e),
                    'failed': True
                })
        
        # Phase 2: LLM Voting
        print(f"\n{Colors.BOLD}🗳️ PHASE 2: LLM CONSENSUS VOTING{Colors.RESET}")
        print("-" * 40)
        
        # Create proposal summary
        successful_proposals = [p for p in proposals if 'failed' not in p]
        if successful_proposals:
            best_proposal = max(successful_proposals, key=lambda x: x.get('score', 0))
            proposal_summary = f"Best proposal: {best_proposal.get('node_id', 'unknown')} (score: {best_proposal.get('score', 0):.2f}) from {best_proposal.get('agent', 'unknown')}"
        else:
            proposal_summary = "No successful proposals received"
        
        votes = []
        for agent in self.agents:
            try:
                print(f"\n{Colors.CYAN}🗳️ {agent.name} voting on consensus...{Colors.RESET}")
                
                vote = agent.vote_on_consensus(proposal_summary)
                votes.append(vote)
                
                # Display vote result
                if 'error' not in vote:
                    vote_decision = vote.get('vote', 'unknown')
                    confidence = vote.get('confidence', 0)
                    reasoning = vote.get('reasoning', 'No reasoning')[:60]
                    
                    vote_emoji = "👍" if vote_decision == 'accept' else "👎"
                    print(f"{Colors.GREEN}{vote_emoji} Vote: {vote_decision} (confidence: {confidence:.2f}) - {reasoning}...{Colors.RESET}")
                else:
                    print(f"{Colors.YELLOW}⚠️ Vote had issues but was handled{Colors.RESET}")
                    
            except Exception as e:
                print(f"{Colors.RED}❌ {agent.name} vote failed: {str(e)[:50]}...{Colors.RESET}")
                votes.append({
                    'agent': agent.name,
                    'error': str(e),
                    'failed': True
                })
        
        # Phase 3: Consensus Evaluation
        print(f"\n{Colors.BOLD}📊 PHASE 3: CONSENSUS EVALUATION{Colors.RESET}")
        print("-" * 40)
        
        successful_votes = [v for v in votes if 'failed' not in v and 'error' not in v]
        total_weight = sum(v.get('weight', 0) for v in successful_votes)
        accept_votes = [v for v in successful_votes if v.get('vote') == 'accept']
        accept_weight = sum(v.get('weight', 0) for v in accept_votes)
        
        # BFT threshold (2/3 majority)
        total_possible_weight = sum(a.weight for a in self.agents)
        required_threshold = (2/3) * total_possible_weight
        
        consensus_achieved = accept_weight >= required_threshold
        
        print(f"  📊 Total possible weight: {total_possible_weight:.1f}")
        print(f"  📊 Voting weight received: {total_weight:.1f}")
        print(f"  📊 Accept weight: {accept_weight:.1f}")
        print(f"  📊 Required threshold (2/3): {required_threshold:.1f}")
        print(f"  📊 Failed agents: {len(self.agents) - len(successful_votes)}")
        
        if consensus_achieved:
            print(f"  {Colors.GREEN}{Colors.BOLD}✅ CONSENSUS ACHIEVED!{Colors.RESET}")
            result = "SUCCESS"
        else:
            print(f"  {Colors.RED}{Colors.BOLD}❌ CONSENSUS FAILED{Colors.RESET}")
            result = "FAILED"
        
        return {
            'job': job_description,
            'result': result,
            'proposals': proposals,
            'votes': votes,
            'accept_weight': accept_weight,
            'required_threshold': required_threshold,
            'consensus_achieved': consensus_achieved
        }
    
    def run_full_interactive_demo(self):
        """Run the complete interactive demonstration"""
        
        print(f"{Colors.BOLD}{Colors.CYAN}")
        print("🧠 INTERACTIVE LLM FAULT TOLERANCE DEMONSTRATION")
        print("=" * 70)
        print(f"{Colors.RESET}")
        
        print("This demo shows REAL LLM interactions with:")
        print("• Actual prompts sent to SambaNova")
        print("• Full LLM responses received")
        print("• Byzantine attack simulations")
        print("• Fault injection and recovery")
        print("• Complete consensus protocol execution")
        print()
        
        # Step 1: Healthy system baseline
        print(f"{Colors.BOLD}🏥 STEP 1: HEALTHY SYSTEM DEMONSTRATION{Colors.RESET}")
        self.display_system_status()
        
        healthy_result = self.run_llm_consensus_demo("AI Training Job: Requires 4 nodes with GPU support for deep learning model training")
        
        print(f"\n{Colors.GREEN}✅ Baseline Result: {healthy_result['result']}{Colors.RESET}")
        
        # Step 2: Fault injection
        print(f"\n{Colors.BOLD}💥 STEP 2: FAULT INJECTION{Colors.RESET}")
        print("-" * 30)
        
        # Inject faults into 2 agents
        faults = [
            ('byzantine', 'Beta-Memory'),
            ('crash', 'Delta-Storage')
        ]
        
        for fault_type, agent_name in faults:
            agent = next(a for a in self.agents if a.name == agent_name)
            agent.inject_fault(fault_type, duration=30)
        
        time.sleep(2)
        self.display_system_status()
        
        # Step 3: Consensus under faults
        print(f"\n{Colors.BOLD}🛡️ STEP 3: CONSENSUS UNDER FAULTS{Colors.RESET}")
        
        fault_result = self.run_llm_consensus_demo("Data Processing Job: Large-scale data analysis requiring high memory nodes")
        
        print(f"\n{Colors.YELLOW}⚠️ Under Faults Result: {fault_result['result']}{Colors.RESET}")
        
        # Step 4: Recovery monitoring
        print(f"\n{Colors.BOLD}🔄 STEP 4: RECOVERY MONITORING{Colors.RESET}")
        print("Waiting for automatic recovery...")
        
        recovery_time = 0
        while recovery_time < 35:
            recovered = sum(1 for agent in self.agents if agent.check_recovery())
            if all(agent.is_healthy for agent in self.agents):
                print(f"  {Colors.GREEN}✅ All agents recovered after {recovery_time}s{Colors.RESET}")
                break
            time.sleep(1)
            recovery_time += 1
        
        self.display_system_status()
        
        # Step 5: Post-recovery verification
        print(f"\n{Colors.BOLD}✨ STEP 5: POST-RECOVERY VERIFICATION{Colors.RESET}")
        
        recovery_result = self.run_llm_consensus_demo("ML Training Job: Multi-GPU machine learning pipeline requiring coordinated resources")
        
        print(f"\n{Colors.GREEN}✅ Post-Recovery Result: {recovery_result['result']}{Colors.RESET}")
        
        # Final summary
        print(f"\n{Colors.BOLD}{Colors.CYAN}📋 DEMONSTRATION SUMMARY{Colors.RESET}")
        print("=" * 50)
        
        results = [healthy_result, fault_result, recovery_result]
        success_count = sum(1 for r in results if r['result'] == 'SUCCESS')
        
        print(f"Scenarios tested: {len(results)}")
        print(f"Successful consensus: {success_count}")
        print(f"Success rate: {success_count/len(results)*100:.1f}%")
        print(f"Total LLM interactions: {sum(len(a.interaction_log) for a in self.agents)}")
        
        print(f"\n{Colors.GREEN}🎉 Interactive LLM demonstration completed!{Colors.RESET}")
        print("All prompts, responses, and reasoning were shown in real-time.")

def main():
    """Main execution function"""
    
    # Validate environment
    if not os.getenv('SAMBASTUDIO_API_KEY') or not os.getenv('SAMBASTUDIO_URL'):
        print(f"{Colors.RED}❌ SambaNova API credentials not found!{Colors.RESET}")
        print("Please set SAMBASTUDIO_API_KEY and SAMBASTUDIO_URL environment variables")
        return
    
    try:
        system = InteractiveFaultTolerantSystem()
        system.run_full_interactive_demo()
        
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Demo interrupted by user{Colors.RESET}")
    except Exception as e:
        print(f"\n{Colors.RED}Demo failed: {e}{Colors.RESET}")
        raise

if __name__ == "__main__":
    main()
