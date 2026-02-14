import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import expm
from typing import List, Dict, Tuple
import seaborn as sns

class MarkovRiskEngine:
    """
    A Markov Chain-based risk engine for quantitative analysis
    """
    
    def __init__(self, states: List[str], transition_matrix: np.ndarray):
        """
        Initialize the Markov Risk Engine
        
        Parameters:
        -----------
        states : List[str]
            List of state names
        transition_matrix : np.ndarray
            Transition probability matrix (n x n)
        """
        self.states = states
        self.n_states = len(states)
        self.transition_matrix = np.array(transition_matrix)
        
        # Validate transition matrix
        self._validate_transition_matrix()
        
    def _validate_transition_matrix(self):
        """Ensure transition matrix is valid (rows sum to 1)"""
        if self.transition_matrix.shape != (self.n_states, self.n_states):
            raise ValueError("Transition matrix dimensions don't match number of states")
        
        row_sums = self.transition_matrix.sum(axis=1)
        if not np.allclose(row_sums, 1.0):
            raise ValueError("Transition matrix rows must sum to 1")
    
    def steady_state(self) -> np.ndarray:
        """
        Calculate steady-state (equilibrium) distribution
        
        Returns:
        --------
        np.ndarray : Steady-state probability distribution
        """
        eigenvalues, eigenvectors = np.linalg.eig(self.transition_matrix.T)
        
        # Find eigenvector corresponding to eigenvalue 1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        steady_state = np.real(eigenvectors[:, idx])
        steady_state = steady_state / steady_state.sum()
        
        return steady_state
    
    def n_step_transition(self, n: int) -> np.ndarray:
        """
        Calculate n-step transition matrix
        
        Parameters:
        -----------
        n : int
            Number of steps
            
        Returns:
        --------
        np.ndarray : n-step transition matrix
        """
        return np.linalg.matrix_power(self.transition_matrix, n)
    
    def simulate_path(self, initial_state: int, n_steps: int, n_simulations: int = 1) -> np.ndarray:
        """
        Simulate Markov chain paths
        
        Parameters:
        -----------
        initial_state : int
            Starting state index
        n_steps : int
            Number of time steps
        n_simulations : int
            Number of simulation paths
            
        Returns:
        --------
        np.ndarray : Simulated paths (n_simulations x n_steps)
        """
        paths = np.zeros((n_simulations, n_steps), dtype=int)
        paths[:, 0] = initial_state
        
        for sim in range(n_simulations):
            current_state = initial_state
            for step in range(1, n_steps):
                # Sample next state based on transition probabilities
                current_state = np.random.choice(
                    self.n_states, 
                    p=self.transition_matrix[current_state]
                )
                paths[sim, step] = current_state
        
        return paths
    
    def calculate_var(self, initial_state: int, loss_values: np.ndarray, 
                      n_steps: int, confidence_level: float = 0.95, 
                      n_simulations: int = 10000) -> Dict:
        """
        Calculate Value at Risk using Monte Carlo simulation
        
        Parameters:
        -----------
        initial_state : int
            Starting state
        loss_values : np.ndarray
            Loss associated with each state
        n_steps : int
            Time horizon
        confidence_level : float
            Confidence level for VaR (default 0.95)
        n_simulations : int
            Number of Monte Carlo simulations
            
        Returns:
        --------
        Dict : VaR metrics including VaR, CVaR, and loss distribution
        """
        # Simulate paths
        paths = self.simulate_path(initial_state, n_steps, n_simulations)
        
        # Calculate cumulative losses for each path
        cumulative_losses = np.zeros(n_simulations)
        for sim in range(n_simulations):
            cumulative_losses[sim] = loss_values[paths[sim]].sum()
        
        # Calculate VaR
        var = np.percentile(cumulative_losses, confidence_level * 100)
        
        # Calculate CVaR (Expected Shortfall)
        cvar = cumulative_losses[cumulative_losses >= var].mean()
        
        return {
            'VaR': var,
            'CVaR': cvar,
            'mean_loss': cumulative_losses.mean(),
            'std_loss': cumulative_losses.std(),
            'loss_distribution': cumulative_losses,
            'confidence_level': confidence_level
        }
    
    def expected_time_to_absorption(self, absorbing_states: List[int]) -> np.ndarray:
        """
        Calculate expected time to reach absorbing states
        
        Parameters:
        -----------
        absorbing_states : List[int]
            Indices of absorbing states
            
        Returns:
        --------
        np.ndarray : Expected time to absorption from each transient state
        """
        # Separate transient and absorbing states
        transient_states = [i for i in range(self.n_states) if i not in absorbing_states]
        
        # Extract Q matrix (transient to transient transitions)
        Q = self.transition_matrix[np.ix_(transient_states, transient_states)]
        
        # Calculate fundamental matrix N = (I - Q)^(-1)
        I = np.eye(len(transient_states))
        N = np.linalg.inv(I - Q)
        
        # Expected time to absorption
        expected_times = N.sum(axis=1)
        
        # Create full result array
        result = np.zeros(self.n_states)
        for i, state in enumerate(transient_states):
            result[state] = expected_times[i]
        
        return result
    
    def visualize_transition_matrix(self, figsize: Tuple = (10, 8)):
        """Visualize the transition matrix as a heatmap"""
        plt.figure(figsize=figsize)
        sns.heatmap(self.transition_matrix, annot=True, fmt='.3f', 
                    xticklabels=self.states, yticklabels=self.states,
                    cmap='YlOrRd', cbar_kws={'label': 'Transition Probability'})
        plt.title('Markov Chain Transition Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('To State', fontsize=12)
        plt.ylabel('From State', fontsize=12)
        plt.tight_layout()
        plt.show()
    
    def visualize_state_evolution(self, initial_distribution: np.ndarray, 
                                   n_steps: int, figsize: Tuple = (12, 6)):
        """Visualize how state probabilities evolve over time"""
        evolution = np.zeros((n_steps, self.n_states))
        evolution[0] = initial_distribution
        
        for t in range(1, n_steps):
            evolution[t] = evolution[t-1] @ self.transition_matrix
        
        plt.figure(figsize=figsize)
        for i, state in enumerate(self.states):
            plt.plot(evolution[:, i], label=state, linewidth=2, marker='o', markersize=4)
        
        plt.xlabel('Time Step', fontsize=12)
        plt.ylabel('Probability', fontsize=12)
        plt.title('State Probability Evolution', fontsize=16, fontweight='bold')
        plt.legend(loc='best')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# Example Usage: Credit Risk Model
def credit_risk_example():
    """
    Example: Credit rating transition model
    States: AAA, AA, A, BBB, BB, B, CCC, Default
    """
    
    # Define states
    states = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC', 'Default']
    
    # Transition matrix (simplified example - 1 year transitions)
    # Based on typical credit rating migration patterns
    transition_matrix = np.array([
        [0.90, 0.08, 0.01, 0.005, 0.003, 0.001, 0.0005, 0.0005],  # AAA
        [0.02, 0.88, 0.08, 0.01, 0.005, 0.003, 0.001, 0.001],      # AA
        [0.005, 0.03, 0.87, 0.07, 0.01, 0.005, 0.005, 0.005],      # A
        [0.002, 0.005, 0.05, 0.85, 0.06, 0.02, 0.01, 0.003],       # BBB
        [0.001, 0.002, 0.01, 0.05, 0.80, 0.10, 0.03, 0.007],       # BB
        [0.0005, 0.001, 0.005, 0.02, 0.06, 0.75, 0.15, 0.0135],    # B
        [0.0002, 0.0005, 0.002, 0.01, 0.03, 0.10, 0.70, 0.1573],   # CCC
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]                   # Default (absorbing)
    ])
    
    # Create risk engine
    engine = MarkovRiskEngine(states, transition_matrix)
    
    print("=" * 70)
    print("MARKOV CHAIN CREDIT RISK ENGINE")
    print("=" * 70)
    
    # 1. Steady state analysis
    steady = engine.steady_state()
    print("\n1. STEADY-STATE DISTRIBUTION:")
    print("-" * 70)
    for state, prob in zip(states, steady):
        print(f"   {state:8s}: {prob:.6f} ({prob*100:.4f}%)")
    
    # 2. Multi-step transitions
    print("\n2. 5-YEAR TRANSITION PROBABILITIES (from BBB):")
    print("-" * 70)
    five_year = engine.n_step_transition(5)
    bbb_index = 3
    for state, prob in zip(states, five_year[bbb_index]):
        print(f"   BBB → {state:8s}: {prob:.6f} ({prob*100:.4f}%)")
    
    # 3. Expected time to default
    print("\n3. EXPECTED TIME TO DEFAULT (years):")
    print("-" * 70)
    absorbing_states = [7]  # Default state
    expected_times = engine.expected_time_to_absorption(absorbing_states)
    for state, time in zip(states[:-1], expected_times[:-1]):
        if time > 0:
            print(f"   {state:8s}: {time:.2f} years")
    
    # 4. VaR calculation
    print("\n4. VALUE AT RISK ANALYSIS:")
    print("-" * 70)
    # Loss values for each state (in basis points of notional)
    loss_values = np.array([0, 50, 100, 200, 500, 1000, 2000, 10000])
    
    var_results = engine.calculate_var(
        initial_state=3,  # Starting from BBB
        loss_values=loss_values,
        n_steps=5,  # 5-year horizon
        confidence_level=0.95,
        n_simulations=10000
    )
    
    print(f"   Starting State: BBB")
    print(f"   Time Horizon: 5 years")
    print(f"   Confidence Level: {var_results['confidence_level']*100}%")
    print(f"   VaR (95%): {var_results['VaR']:.2f} bps")
    print(f"   CVaR (Expected Shortfall): {var_results['CVaR']:.2f} bps")
    print(f"   Mean Loss: {var_results['mean_loss']:.2f} bps")
    print(f"   Std Dev: {var_results['std_loss']:.2f} bps")
    
    # 5. Visualizations
    print("\n5. GENERATING VISUALIZATIONS...")
    print("-" * 70)
    
    # Transition matrix heatmap
    engine.visualize_transition_matrix()
    
    # State evolution from BBB rating
    initial_dist = np.zeros(len(states))
    initial_dist[3] = 1.0  # Start at BBB
    engine.visualize_state_evolution(initial_dist, n_steps=20)
    
    # Loss distribution histogram
    plt.figure(figsize=(12, 6))
    plt.hist(var_results['loss_distribution'], bins=50, alpha=0.7, 
             edgecolor='black', color='steelblue')
    plt.axvline(var_results['VaR'], color='red', linestyle='--', 
                linewidth=2, label=f"VaR (95%): {var_results['VaR']:.0f} bps")
    plt.axvline(var_results['CVaR'], color='darkred', linestyle='--', 
                linewidth=2, label=f"CVaR: {var_results['CVaR']:.0f} bps")
    plt.xlabel('Cumulative Loss (bps)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Loss Distribution (5-year horizon from BBB)', 
              fontsize=16, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)

# Run the example
if __name__ == "__main__":
    credit_risk_example()