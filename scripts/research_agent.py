import os
import sys
import json
import time
import re
import signal
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add project root to path for backend imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from backend.app.utils.llm_client import LLMClient
    from duckduckgo_search import DDGS
except ImportError:
    print("Dependencies missing. Ensure 'openai' and 'duckduckgo_search' are installed.")
    sys.exit(1)

class ResearchAgent:
    """Deep Research Agent for MiroFish."""
    
    def __init__(self, api_key: str, model: str = "moonshotai/kimi-k2-instruct"):
        # Groq API base URL
        self.llm = LLMClient(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            model=model
        )
        self.search_tool = DDGS()
        self.results_dir = os.path.join(os.path.dirname(__file__), '../results')
        self.findings = []
        self.research_name = "Untitled Research"
        self.start_time = time.time()
        self.max_duration = 3300  # 55 minutes (to allow time for report generation)
        self.interrupted = False
        
        # Setup signal handler for graceful shutdown (if possible)
        signal.signal(signal.SIGINT, self._handle_interrupt)
        signal.signal(signal.SIGTERM, self._handle_interrupt)

    def _handle_interrupt(self, signum, frame):
        print(f"\n[!] Interrupt received (signal {signum}). Finalizing research...")
        self.interrupted = True

    def _get_elapsed_time(self) -> float:
        return time.time() - self.start_time

    def _parse_queries(self, text: str) -> List[str]:
        """Extract search queries from LLM response."""
        queries = []
        # Look for <search>query</search> or just a list
        matches = re.findall(r'<search>(.*?)</search>', text, re.DOTALL)
        if matches:
            queries = [q.strip() for q in matches]
        else:
            # Fallback: look for numbered list or lines
            lines = text.split('\n')
            for line in lines:
                if line.strip().startswith(('- ', '* ', '1. ', '2. ', '3. ')):
                    query = re.sub(r'^[-*0-9.]+ ', '', line).strip()
                    if query and len(query) < 100:
                        queries.append(query)
        return queries[:5]  # Limit queries per step

    def search(self, query: str) -> str:
        """Perform search with DuckDuckGo."""
        try:
            print(f"[*] Searching: {query}")
            results = list(self.search_tool.text(query, max_results=8))
            if not results:
                return "No results found."
            
            formatted = []
            for i, res in enumerate(results, 1):
                formatted.append(f"Title: {res.get('title')}\nLink: {res.get('href')}\nSnippet: {res.get('body')}")
            
            return "\n\n".join(formatted)
        except Exception as e:
            return f"Search error: {str(e)}"

    def deep_research(self, question: str):
        """Main research loop."""
        self.research_name = question[:50].replace(" ", "_").strip("?")
        # Sanitize filename
        self.research_name = re.sub(r'[^\w\-_]', '_', self.research_name)
        print(f"[*] Starting Deep Research: {question}")
        
        # Initial thought and queries
        messages = [
            {"role": "system", "content": (
                "You are the MiroFish Deep Research Agent. Your goal is to conduct exhaustive research on the user's question.\n"
                "You will iterate through multiple rounds of searching and analysis.\n"
                "In each round, provide a brief 'Thought' and then list up to 5 search queries inside <search>...</search> tags.\n"
                "Example:\n"
                "Thought: I need to find the latest breakthroughs in swarm intelligence.\n"
                "<search>latest breakthroughs swarm intelligence 2025</search>\n"
                "<search>multi-agent simulation trends 2026</search>\n"
                "\n"
                "Current Date: " + datetime.now().strftime("%Y-%m-%d") + "\n"
                "RPD Limit: 1000. Use tokens efficiently (low TPM)."
            )},
            {"role": "user", "content": f"Research Question: {question}"}
        ]

        iteration = 0
        while not self.interrupted and self._get_elapsed_time() < self.max_duration:
            iteration += 1
            print(f"\n[Round {iteration}] Thinking...")
            
            try:
                response = self.llm.chat(messages, temperature=0.3, max_tokens=1000)
                print(f"[LLM] {response[:200]}...")
                
                messages.append({"role": "assistant", "content": response})
                
                queries = self._parse_queries(response)
                if not queries:
                    # If no queries, ask LLM to provide some or synthesize
                    messages.append({"role": "user", "content": "Please provide specific search queries in <search> tags to continue research."})
                    continue

                observations = []
                for query in queries:
                    if self.interrupted or self._get_elapsed_time() >= self.max_duration:
                        break
                    result = self.search(query)
                    observations.append(f"Query: {query}\nResult:\n{result}")
                    # Rate limiting to stay under TPM/RPM
                    time.sleep(2) 

                if observations:
                    obs_text = "\n\n---\n\n".join(observations)
                    self.findings.append(obs_text)
                    messages.append({"role": "user", "content": f"Observations from searches:\n{obs_text[:2000]}..."}) # Truncate for TPM
                
                # Save progress periodically
                self._save_progress(question)

            except Exception as e:
                print(f"[!] Error in loop: {e}")
                time.sleep(10)
                continue

        print("\n[*] Research phase complete. Generating report...")
        self.generate_report(question)

    def _save_progress(self, question: str):
        """Save raw findings to a progress file."""
        date_str = datetime.now().strftime("%Y-%m-%d")
        folder_path = os.path.join(self.results_dir, date_str, "progress")
        os.makedirs(folder_path, exist_ok=True)
        
        progress_file = os.path.join(folder_path, f"{self.research_name}_progress.txt")
        with open(progress_file, 'w', encoding='utf-8') as f:
            f.write(f"Research Question: {question}\n")
            f.write(f"Status: In Progress\n")
            f.write(f"Last Update: {datetime.now().isoformat()}\n\n")
            f.write("\n\n---\n\n".join(self.findings))

    def generate_report(self, question: str):
        """Synthesize all findings into a markdown report."""
        print("[*] Synthesizing findings...")
        
        # We might have a lot of findings. Let's summarize them if too large.
        summary_context = "\n\n".join(self.findings[-10:]) # Take last few for final touch
        
        report_prompt = [
            {"role": "system", "content": "You are a professional research analyst. Create a comprehensive, detailed, and structured research report based on the findings provided."},
            {"role": "user", "content": f"Question: {question}\n\nFindings context (summarized):\n{summary_context[:5000]}\n\nGenerate a full markdown report with Introduction, Key Breakthroughs, Detailed Analysis, Future Outlook, and References."}
        ]
        
        try:
            report_content = self.llm.chat(report_prompt, temperature=0.5, max_tokens=4000)
            
            # Save report
            date_str = datetime.now().strftime("%Y-%m-%d")
            folder_path = os.path.join(self.results_dir, date_str)
            os.makedirs(folder_path, exist_ok=True)
            
            filename = f"{self.research_name}.md"
            file_path = os.path.join(folder_path, filename)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"# Research Report: {question}\n")
                f.write(f"**Date:** {date_str}\n")
                f.write(f"**Model:** moonshotai/kimi-k2-instruct\n\n")
                f.write(report_content)
            
            print(f"[+] Report saved to {file_path}")
            
        except Exception as e:
            print(f"[!] Error generating report: {e}")
            # Save raw findings if report fails
            date_str = datetime.now().strftime("%Y-%m-%d")
            folder_path = os.path.join(self.results_dir, date_str)
            os.makedirs(folder_path, exist_ok=True)
            with open(os.path.join(folder_path, f"{self.research_name}_raw.txt"), 'w') as f:
                f.write("\n\n".join(self.findings))

def main():
    api_key = os.environ.get("GROQ")
    if not api_key:
        print("[!] Missing GROQ environment variable.")
        return

    # Read question from README.md
    readme_path = os.path.join(os.path.dirname(__file__), '../README.md')
    question = "What are the latest breakthroughs in swarm intelligence and multi-agent simulation in 2025 and 2026?"
    
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read()
            match = re.search(r'### 📝 Current Question\s*\n\s*\*\*(.*?)\*\*', content, re.DOTALL)
            if match:
                question = match.group(1).strip()

    agent = ResearchAgent(api_key=api_key)
    agent.deep_research(question)

if __name__ == "__main__":
    main()
