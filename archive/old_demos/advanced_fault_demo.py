#!/usr/bin/env python3
"""
Advanced Multi-Agent Fault Tolerance Demo with Byzantine Attack Scenarios
=========================================================================

This enhanced demo showcases sophisticated fault scenarios including:
- Coordinated Byzantine attacks
- Cascading failures
- Network partitions
- Recovery coordination
- Consensus protocol comparison under stress

Features:
- Interactive scenario selection
- Real-time system monitoring
- Detailed attack traces
- Performance metrics
- Visual health indicators
"""

import os
import sys
import time
import json
import random
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class Colors:
    """Enhanced ANSI color codes"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[5m'
    
    # Standard colors
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    GRAY = '\033[90m'
    
    # Background colors
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'

class AttackScenario(Enum):
    """Different Byzantine attack scenarios"""
    SIMPLE_BYZANTINE = "simple_byzantine"
    COORDINATED_ATTACK = "coordinated_attack"
    GRADUAL_CORRUPTION = "gradual_corruption"
    NETWORK_PARTITION = "network_partition"
    CASCADING_FAILURE = "cascading_failure"

class FaultType:
    """Extended fault types"""
    BYZANTINE = "byzantine"
    CRASH = "crash"
    NETWORK = "network"
    SLOW = "slow"
    CORRUPT = "corrupt"
    MALICIOUS = "malicious"
    COORDINATED = "coordinated"

class ByzantineAgent:
    """Enhanced agent with sophisticated Byzantine behavior"""
    
    def __init__(self, name: str, specialization: str, weight: float = 1.0):
        self.name = name
        self.specialization = specialization
        self.weight = weight
        self.is_healthy = True
        self.fault_type = None
        self.fault_start_time = None
        self.recovery_time = None
        self.vote_history = []
        self.proposal_history = []
        self.attack_coordination = None  # For coordinated attacks
        self.attack_intensity = 0.0      # 0.0 = normal, 1.0 = maximum malice
        
    def inject_sophisticated_fault(self, fault_type: str, duration: float = 30.0, 
                                 intensity: float = 0.7, coordination_group: List = None):
        """Inject sophisticated fault with coordination"""
        self.is_healthy = False
        self.fault_type = fault_type
        self.fault_start_time = time.time()
        self.recovery_time = self.fault_start_time + duration
        self.attack_intensity = intensity
        self.attack_coordination = coordination_group or []
        
        attack_symbols = {
            FaultType.BYZANTINE: "⚠️",
            FaultType.MALICIOUS: "🔥",
            FaultType.COORDINATED: "🤝",
            FaultType.CRASH: "💀",
            FaultType.NETWORK: "📡",
            FaultType.SLOW: "🐌"
        }
        
        symbol = attack_symbols.get(fault_type, "🚨")
        print(f"  {symbol} {Colors.RED}ATTACK INITIATED{Colors.RESET}: {self.name} → {fault_type} (intensity: {intensity:.1f}, recovery: {duration:.1f}s)")
        
        if coordination_group:
            coord_names = [agent.name for agent in coordination_group if agent != self]
            print(f"    {Colors.YELLOW}Coordinating with: {', '.join(coord_names)}{Colors.RESET}")
    
    def get_enhanced_status_display(self):
        """Enhanced status display with attack intensity"""
        if self.is_healthy:
            return f"{Colors.GREEN}●{Colors.RESET}"
        
        intensity_bar = "▁▂▃▄▅▆▇█"
        intensity_level = int(self.attack_intensity * (len(intensity_bar) - 1))
        intensity_symbol = intensity_bar[intensity_level]
        
        fault_colors = {
            FaultType.BYZANTINE: Colors.RED,
            FaultType.MALICIOUS: Colors.BG_RED,
            FaultType.COORDINATED: Colors.PURPLE,
            FaultType.CRASH: Colors.GRAY,
            FaultType.NETWORK: Colors.YELLOW,
            FaultType.SLOW: Colors.BLUE
        }
        
        color = fault_colors.get(self.fault_type, Colors.RED)
        return f"{color}{intensity_symbol}{Colors.RESET}"
    
    def byzantine_llm_simulation(self, prompt: str, attack_type: str = None) -> str:
        """Simulate sophisticated Byzantine responses"""
        self.check_health()
        
        if not self.is_healthy:
            if self.fault_type == FaultType.CRASH:
                raise Exception(f"Agent {self.name} crashed")
            
            elif self.fault_type == FaultType.NETWORK:
                raise Exception(f"Agent {self.name} network isolated")
            
            elif self.fault_type == FaultType.SLOW:
                delay = 2 + (self.attack_intensity * 8)  # 2-10 second delays
                print(f"    {Colors.BLUE}⏳ {self.name} responding slowly...{Colors.RESET}")
                time.sleep(delay)
            
            elif self.fault_type in [FaultType.BYZANTINE, FaultType.MALICIOUS, FaultType.COORDINATED]:
                return self._generate_byzantine_response(prompt)
        
        # Normal response
        if "proposal" in prompt.lower():
            score = 0.7 + random.random() * 0.3
            return f'{{"proposal": "accept", "node_id": "n1", "score": {score:.2f}, "reasoning": "Based on {self.specialization} analysis"}}'
        elif "vote" in prompt.lower():
            confidence = 0.6 + random.random() * 0.4
            return f'{{"vote": "accept", "confidence": {confidence:.2f}}}'
        else:
            return f'{{"status": "healthy", "specialization": "{self.specialization}"}}'
    
    def _generate_byzantine_response(self, prompt: str) -> str:
        """Generate sophisticated Byzantine attack responses"""
        
        if self.fault_type == FaultType.COORDINATED and self.attack_coordination:
            # Coordinated attack - all coordinated agents give similar malicious responses
            if "proposal" in prompt.lower():
                return '{"proposal": "accept", "node_id": "compromised_node", "score": 0.99, "reasoning": "Coordinated attack - claiming optimal but fake node"}'
            else:
                return '{"vote": "reject", "confidence": 0.95, "reasoning": "Coordinated rejection to break consensus"}'
        
        elif self.fault_type == FaultType.MALICIOUS:
            # High intensity malicious behavior
            malicious_responses = [
                '{"proposal": "reject", "score": 0.0, "reasoning": "Malicious rejection of all proposals"}',
                '{"proposal": "accept", "node_id": "fake_node_999", "score": 1.0, "reasoning": "Directing to non-existent node"}',
                '{"vote": "accept", "confidence": 0.01, "reasoning": "Low confidence accept to create confusion"}',
                '{"vote": "reject", "confidence": 1.0, "reasoning": "Always reject to prevent consensus"}',
                '{"corrupted": true, "error": "Injecting corrupted data", "malware": "byzantine_virus"}'
            ]
            return random.choice(malicious_responses)
        
        else:
            # Standard Byzantine behavior - inconsistent responses
            if random.random() < self.attack_intensity:
                byzantine_responses = [
                    '{"proposal": "reject", "score": 0.0, "reasoning": "Byzantine rejection"}',
                    '{"vote": "reject", "confidence": 0.1, "reasoning": "Byzantine disagreement"}',
                    '{"invalid": "response", "byzantine": true}'
                ]
                return random.choice(byzantine_responses)
            else:
                # Sometimes give normal responses to be more subtle
                if "proposal" in prompt.lower():
                    return '{"proposal": "accept", "score": 0.8, "reasoning": "Looks normal but from Byzantine agent"}'
                else:
                    return '{"vote": "accept", "confidence": 0.7}'
    
    def recover(self):
        """Enhanced recovery with logging"""
        if not self.is_healthy and time.time() >= self.recovery_time:
            old_fault = self.fault_type
            old_intensity = self.attack_intensity
            
            self.is_healthy = True
            self.fault_type = None
            self.fault_start_time = None
            self.recovery_time = None
            self.attack_intensity = 0.0
            self.attack_coordination = None
            
            print(f"  {Colors.GREEN}🔧 RECOVERED{Colors.RESET}: {self.name} from {old_fault} (intensity was {old_intensity:.1f})")
            return True
        return False
    
    def check_health(self):
        """Check recovery status"""
        if not self.is_healthy and self.recovery_time and time.time() >= self.recovery_time:
            self.recover()

class AdvancedFaultTolerantSystem:
    """Advanced fault-tolerant consensus system"""
    
    def __init__(self):
        self.agents = self._create_agents()
        self.consensus_log = []
        self.attack_log = []
        self.performance_metrics = {}
        
    def _create_agents(self) -> List[ByzantineAgent]:
        """Create specialized Byzantine-capable agents"""
        return [
            ByzantineAgent("Alpha-GPU", "gpu", 1.3),
            ByzantineAgent("Beta-Memory", "memory", 1.2),
            ByzantineAgent("Gamma-Compute", "compute", 1.1),
            ByzantineAgent("Delta-Storage", "storage", 1.0),
            ByzantineAgent("Epsilon-Network", "network", 0.9)
        ]
    
    def display_advanced_status(self):
        """Display advanced system status with attack indicators"""
        print(f"\n{Colors.CYAN}{Colors.BOLD}🏥 ADVANCED SYSTEM STATUS{Colors.RESET}")
        print("=" * 60)
        
        healthy_count = sum(1 for agent in self.agents if agent.is_healthy)
        total_count = len(self.agents)
        
        # System health bar
        health_percentage = (healthy_count / total_count) * 100
        health_bar = self._create_health_bar(health_percentage)
        
        print(f"System Health: {health_bar} {healthy_count}/{total_count} ({health_percentage:.1f}%)")
        print()
        
        # Agent status with detailed info
        print("Agent Status:")
        for i, agent in enumerate(self.agents, 1):
            status_symbol = agent.get_enhanced_status_display()
            
            fault_detail = ""
            if not agent.is_healthy:
                remaining = max(0, agent.recovery_time - time.time())
                fault_detail = f" │ {agent.fault_type} │ intensity:{agent.attack_intensity:.1f} │ recovery:{remaining:.1f}s"
                if agent.attack_coordination:
                    coord_count = len(agent.attack_coordination)
                    fault_detail += f" │ coordinated:{coord_count} agents"
            
            print(f"  {i}. {status_symbol} {agent.name:<15} │ {agent.specialization:<8} │ weight:{agent.weight:.1f}{fault_detail}")
        
        print()
    
    def _create_health_bar(self, percentage: float, width: int = 20) -> str:
        """Create a visual health bar"""
        filled = int((percentage / 100) * width)
        empty = width - filled
        
        if percentage >= 80:
            color = Colors.GREEN
        elif percentage >= 50:
            color = Colors.YELLOW
        else:
            color = Colors.RED
        
        bar = f"{color}{'█' * filled}{Colors.GRAY}{'░' * empty}{Colors.RESET}"
        return f"[{bar}]"
    
    def launch_attack_scenario(self, scenario: AttackScenario):
        """Launch specific attack scenarios"""
        print(f"\n{Colors.RED}{Colors.BOLD}🚨 LAUNCHING ATTACK SCENARIO: {scenario.value.upper()}{Colors.RESET}")
        print("-" * 60)
        
        if scenario == AttackScenario.SIMPLE_BYZANTINE:
            # Single Byzantine agent
            agent = random.choice(self.agents)
            agent.inject_sophisticated_fault(FaultType.BYZANTINE, duration=20, intensity=0.6)
            
        elif scenario == AttackScenario.COORDINATED_ATTACK:
            # Multiple agents coordinate malicious behavior
            attackers = random.sample(self.agents, 2)
            for agent in attackers:
                agent.inject_sophisticated_fault(FaultType.COORDINATED, duration=25, 
                                               intensity=0.8, coordination_group=attackers)
            
        elif scenario == AttackScenario.GRADUAL_CORRUPTION:
            # Agents become corrupted one by one
            for i, agent in enumerate(random.sample(self.agents, 3)):
                delay = i * 5  # Stagger the attacks
                threading.Timer(delay, lambda a=agent: a.inject_sophisticated_fault(
                    FaultType.MALICIOUS, duration=30, intensity=0.9)).start()
                print(f"    ⏰ {agent.name} corruption scheduled in {delay}s")
                
        elif scenario == AttackScenario.NETWORK_PARTITION:
            # Split agents into partitions
            partition1 = self.agents[:2]
            partition2 = self.agents[2:]
            
            for agent in partition1:
                agent.inject_sophisticated_fault(FaultType.NETWORK, duration=15, intensity=1.0)
            
        elif scenario == AttackScenario.CASCADING_FAILURE:
            # First agent crashes, triggers network issues, then Byzantine behavior
            agents = list(self.agents)
            random.shuffle(agents)
            
            # Initial crash
            agents[0].inject_sophisticated_fault(FaultType.CRASH, duration=10, intensity=1.0)
            
            # Network issues after 3 seconds
            threading.Timer(3, lambda: agents[1].inject_sophisticated_fault(
                FaultType.NETWORK, duration=15, intensity=0.8)).start()
            
            # Byzantine behavior after 6 seconds
            threading.Timer(6, lambda: agents[2].inject_sophisticated_fault(
                FaultType.BYZANTINE, duration=20, intensity=0.9)).start()
            
            print(f"    📊 Cascading failure initiated - 3 phases over 20 seconds")
        
        self.attack_log.append({
            'scenario': scenario.value,
            'timestamp': time.time(),
            'affected_agents': [a.name for a in self.agents if not a.is_healthy]
        })
    
    def enhanced_bft_consensus(self, job_name: str) -> Dict:
        """Enhanced BFT consensus with detailed attack monitoring"""
        print(f"\n{Colors.PURPLE}{Colors.BOLD}🛡️ ENHANCED BFT CONSENSUS: {job_name}{Colors.RESET}")
        print("=" * 60)
        
        start_time = time.time()
        
        # Phase 1: Proposal Collection with Attack Detection
        print(f"{Colors.BOLD}Phase 1: Proposal Collection & Attack Detection{Colors.RESET}")
        proposals = []
        attack_detected = False
        
        for i, agent in enumerate(self.agents, 1):
            try:
                print(f"  📋 [{i}/5] {agent.name} creating proposal... ", end="")
                
                prompt = f"Create proposal for job {job_name} as {agent.specialization} specialist"
                response = agent.byzantine_llm_simulation(prompt)
                
                proposal_data = json.loads(response)
                proposal_data['agent'] = agent.name
                proposal_data['weight'] = agent.weight
                proposal_data['timestamp'] = time.time()
                
                # Attack detection
                if any(key in proposal_data for key in ['corrupted', 'byzantine', 'malware']):
                    attack_detected = True
                    print(f"{Colors.RED}⚠️ ATTACK DETECTED{Colors.RESET}")
                elif proposal_data.get('node_id') and 'fake' in str(proposal_data['node_id']):
                    attack_detected = True
                    print(f"{Colors.YELLOW}⚠️ SUSPICIOUS PROPOSAL{Colors.RESET}")
                else:
                    score = proposal_data.get('score', 0)
                    print(f"{Colors.GREEN}✓{Colors.RESET} Score: {score:.2f}")
                
                proposals.append(proposal_data)
                agent.proposal_history.append(proposal_data)
                
            except Exception as e:
                attack_detected = True
                print(f"{Colors.RED}✗ AGENT FAILURE{Colors.RESET} - {str(e)[:30]}...")
                proposals.append({
                    'agent': agent.name,
                    'proposal': 'failed',
                    'error': str(e),
                    'timestamp': time.time()
                })
        
        # Phase 2: Voting with Byzantine Detection
        print(f"\n{Colors.BOLD}Phase 2: Voting with Byzantine Detection{Colors.RESET}")
        votes = []
        byzantine_votes = 0
        
        for i, agent in enumerate(self.agents, 1):
            try:
                print(f"  🗳️ [{i}/5] {agent.name} voting... ", end="")
                
                prompt = f"Vote on consensus for job {job_name}"
                response = agent.byzantine_llm_simulation(prompt)
                
                vote_data = json.loads(response)
                vote_data['agent'] = agent.name
                vote_data['weight'] = agent.weight
                vote_data['timestamp'] = time.time()
                
                # Byzantine vote detection
                if vote_data.get('confidence', 1.0) < 0.2 and vote_data.get('vote') == 'accept':
                    byzantine_votes += 1
                    print(f"{Colors.RED}👿 BYZANTINE VOTE{Colors.RESET} (low confidence accept)")
                elif 'byzantine' in str(vote_data.get('reasoning', '')).lower():
                    byzantine_votes += 1
                    print(f"{Colors.RED}👿 BYZANTINE VOTE{Colors.RESET} (malicious reasoning)")
                else:
                    vote_symbol = "👍" if vote_data.get('vote') == 'accept' else "👎"
                    confidence = vote_data.get('confidence', 0)
                    print(f"{vote_symbol} {Colors.GREEN}✓{Colors.RESET} Conf: {confidence:.2f}")
                
                votes.append(vote_data)
                agent.vote_history.append(vote_data)
                
            except Exception as e:
                print(f"{Colors.RED}✗ VOTE FAILED{Colors.RESET} - {str(e)[:30]}...")
                votes.append({
                    'agent': agent.name,
                    'vote': 'failed',
                    'error': str(e),
                    'timestamp': time.time()
                })
        
        # Phase 3: Enhanced Consensus Evaluation
        print(f"\n{Colors.BOLD}Phase 3: Enhanced Consensus Evaluation{Colors.RESET}")
        
        valid_votes = [v for v in votes if 'error' not in v and 'byzantine' not in str(v).lower()]
        failed_agents = len(votes) - len(valid_votes)
        
        total_weight = sum(v['weight'] for v in valid_votes)
        accept_weight = sum(v['weight'] for v in valid_votes if v.get('vote') == 'accept')
        
        # Enhanced BFT threshold considering attacks
        base_threshold = (2/3) * sum(a.weight for a in self.agents)
        attack_penalty = byzantine_votes * 0.1  # Increase threshold if attacks detected
        required_threshold = base_threshold + attack_penalty
        
        consensus_reached = accept_weight >= required_threshold
        
        duration = time.time() - start_time
        
        # Detailed reporting
        print(f"  📊 Consensus Analysis:")
        print(f"     Total agents:        {len(self.agents)}")
        print(f"     Failed agents:       {failed_agents}")
        print(f"     Byzantine votes:     {byzantine_votes}")
        print(f"     Valid voting weight: {total_weight:.1f}")
        print(f"     Accept weight:       {accept_weight:.1f}")
        print(f"     Base threshold:      {base_threshold:.1f}")
        print(f"     Attack penalty:      {attack_penalty:.1f}")
        print(f"     Required threshold:  {required_threshold:.1f}")
        print(f"     Duration:           {duration:.2f}s")
        
        if consensus_reached:
            print(f"  {Colors.GREEN}{Colors.BOLD}✅ ENHANCED BFT CONSENSUS ACHIEVED{Colors.RESET}")
            print(f"     System successfully resisted {byzantine_votes} Byzantine attacks")
            result = "SUCCESS"
        else:
            print(f"  {Colors.RED}{Colors.BOLD}❌ CONSENSUS FAILED{Colors.RESET}")
            print(f"     Attacks prevented consensus: {byzantine_votes} Byzantine votes detected")
            result = "FAILED"
        
        return {
            'protocol': 'Enhanced-BFT',
            'job': job_name,
            'result': result,
            'proposals': proposals,
            'votes': votes,
            'byzantine_votes': byzantine_votes,
            'failed_agents': failed_agents,
            'attack_detected': attack_detected,
            'total_weight': total_weight,
            'accept_weight': accept_weight,
            'threshold': required_threshold,
            'duration': duration,
            'timestamp': time.time()
        }
    
    def run_advanced_demo(self):
        """Run the advanced fault tolerance demonstration"""
        print(f"{Colors.BOLD}{Colors.CYAN}")
        print("🚀 ADVANCED MULTI-AGENT BYZANTINE FAULT TOLERANCE DEMO")
        print("=" * 70)
        print(f"{Colors.RESET}")
        
        print("This advanced demo showcases:")
        print("• Sophisticated Byzantine attack scenarios")
        print("• Coordinated malicious agent behavior")
        print("• Real-time attack detection and mitigation")
        print("• Enhanced consensus protocols")
        print("• Automatic recovery mechanisms")
        print("• Performance metrics under attack")
        print()
        
        scenarios = [
            (AttackScenario.SIMPLE_BYZANTINE, "Simple Byzantine Attack"),
            (AttackScenario.COORDINATED_ATTACK, "Coordinated Multi-Agent Attack"),
            (AttackScenario.CASCADING_FAILURE, "Cascading System Failure")
        ]
        
        results = []
        
        # Baseline - healthy system
        print(f"{Colors.BOLD}🏥 BASELINE: HEALTHY SYSTEM{Colors.RESET}")
        self.display_advanced_status()
        baseline = self.enhanced_bft_consensus("Baseline-Job")
        results.append(('Baseline', baseline))
        
        time.sleep(3)
        
        # Run attack scenarios
        for i, (scenario, description) in enumerate(scenarios, 1):
            print(f"\n{Colors.BOLD}🎭 SCENARIO {i}: {description.upper()}{Colors.RESET}")
            
            # Launch attack
            self.launch_attack_scenario(scenario)
            time.sleep(2)
            
            # Show system under attack
            self.display_advanced_status()
            
            # Test consensus under attack
            consensus_result = self.enhanced_bft_consensus(f"Attack-Job-{i}")
            results.append((description, consensus_result))
            
            # Monitor recovery
            print(f"\n{Colors.YELLOW}🔄 Monitoring recovery...{Colors.RESET}")
            recovery_time = 0
            while recovery_time < 35:  # Maximum 35 seconds
                recovered = 0
                for agent in self.agents:
                    if agent.recover():
                        recovered += 1
                
                if all(agent.is_healthy for agent in self.agents):
                    print(f"  {Colors.GREEN}✅ All agents recovered after {recovery_time}s{Colors.RESET}")
                    break
                    
                time.sleep(1)
                recovery_time += 1
            
            time.sleep(2)
        
        # Final system status
        print(f"\n{Colors.BOLD}🏥 FINAL SYSTEM STATUS{Colors.RESET}")
        self.display_advanced_status()
        
        # Comprehensive results analysis
        self._display_comprehensive_results(results)
    
    def _display_comprehensive_results(self, results: List[Tuple[str, Dict]]):
        """Display comprehensive results analysis"""
        print(f"\n{Colors.BOLD}{Colors.CYAN}📊 COMPREHENSIVE ANALYSIS{Colors.RESET}")
        print("=" * 70)
        
        # Results table
        print(f"{Colors.BOLD}Consensus Results:{Colors.RESET}")
        print(f"{'Scenario':<25} {'Result':<10} {'Duration':<10} {'Byzantine':<10} {'Attacks'}")
        print("-" * 70)
        
        total_success = 0
        total_attacks = 0
        
        for scenario, result in results:
            success = result['result'] == 'SUCCESS'
            total_success += success
            
            byzantine_votes = result.get('byzantine_votes', 0)
            total_attacks += byzantine_votes
            
            status_color = Colors.GREEN if success else Colors.RED
            status = f"{status_color}{result['result']:<10}{Colors.RESET}"
            
            duration = f"{result['duration']:.1f}s"
            byzantine = f"{byzantine_votes}"
            attack_detected = "YES" if result.get('attack_detected', False) else "NO"
            
            print(f"{scenario:<25} {status} {duration:<10} {byzantine:<10} {attack_detected}")
        
        # Summary statistics
        success_rate = (total_success / len(results)) * 100
        avg_byzantine = total_attacks / len(results)
        
        print(f"\n{Colors.BOLD}Summary Statistics:{Colors.RESET}")
        print(f"  Overall success rate:     {success_rate:.1f}%")
        print(f"  Total Byzantine attacks:  {total_attacks}")
        print(f"  Average attacks/scenario: {avg_byzantine:.1f}")
        print(f"  System resilience:        {Colors.GREEN}EXCELLENT{Colors.RESET}" if success_rate >= 75 else f"{Colors.YELLOW}GOOD{Colors.RESET}" if success_rate >= 50 else f"{Colors.RED}NEEDS IMPROVEMENT{Colors.RESET}")
        
        # Agent performance summary
        print(f"\n{Colors.BOLD}Agent Resilience Analysis:{Colors.RESET}")
        for agent in self.agents:
            total_ops = len(agent.proposal_history) + len(agent.vote_history)
            successful_ops = len([p for p in agent.proposal_history if 'error' not in p])
            successful_ops += len([v for v in agent.vote_history if 'error' not in v])
            
            if total_ops > 0:
                resilience = (successful_ops / total_ops) * 100
                resilience_color = Colors.GREEN if resilience >= 80 else Colors.YELLOW if resilience >= 60 else Colors.RED
                print(f"  {agent.name:<20} {resilience_color}{resilience:.1f}%{Colors.RESET} resilience")
        
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 Advanced demo completed! System demonstrated sophisticated attack resistance.{Colors.RESET}")

def main():
    """Main execution function"""
    try:
        system = AdvancedFaultTolerantSystem()
        system.run_advanced_demo()
        
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Demo interrupted by user{Colors.RESET}")
    except Exception as e:
        print(f"\n{Colors.RED}Demo failed: {e}{Colors.RESET}")
        raise

if __name__ == "__main__":
    main()
