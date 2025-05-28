# To run this Locust test:
# 1. Install Locust: pip install locust
# 2. Run the test: locust -f locust_test.py
# 3. Open http://localhost:8089 in your browser
# 4. Set the number of users, spawn rate, and host URL (e.g., https://modal-labs-advay-dev--endpoint-test-model-generate.modal.run)
# 5. Start the test and monitor results in real-time
import csv
import random
import time
from datetime import datetime

from locust import HttpUser, between, task


class ModelUser(HttpUser):
    wait_time = between(1, 5)  # Wait between 1 and 5 seconds between tasks

    # List of LoRA adapters to randomly choose from
    LORA_ADAPTERS = [
        "summaries-fp16",
        "summaries-bf16",
    ]

    # List of sample prompts to randomly choose from
    SAMPLE_PROMPTS = [
        "What are the 3 laws of thermodynamics?",
        "Explain the rules of chess",
        "Who was Albert Einstein?",
        "What is the capital of France?",
        "Explain how photosynthesis works",
        "What is the difference between DNA and RNA?",
        "Describe the water cycle",
        "What is the theory of relativity?",
        "How does a computer processor work?",
        "What are the main causes of climate change?",
        "78 * 23",
        "Explain the anatomy of the human heart",
        "What is the Pythagorean theorem?",
        "How do vaccines work?",
        "Describe the process of evolution",
    ]

    def on_start(self):
        # Create CSV file with timestamp in filename
        timestamp = datetime.now().strftime("%Y%m%d_%H")
        self.csv_filename = f"response_times_simple_{timestamp}.csv"

        # Create and initialize the CSV file with headers
        with open(self.csv_filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["timestamp", "model", "prompt", "status", "response_time_s"])

    @task
    def generate_text(self):
        # Select a random prompt and LoRA adapter
        prompt = random.choice(self.SAMPLE_PROMPTS)
        model = random.choice(self.LORA_ADAPTERS)
        # model = "meta-llama/Llama-3.1-8B-Instruct"

        # Prepare the request payload
        payload = {
            "model": model,
            "prompt": "\nYou will be given a news article. Your task is to summarize the article in a concise manner. Your summary should be no more than 70 words.\n\nNews article:\nWASHINGTON (CNN) -- A pair of tornadoes struck suburban Washington on Sunday, mangling trees and stripping siding off several homes, the National Weather Service confirmed. No injuries were immediately reported. The first tornado hit St. Charles, Maryland -- about 30 miles south of Washington -- just after 2 p.m. It uprooted several trees, many of which fell onto cars and homes. The strongest wind from that touchdown was 80 mph -- enough force to blow out windows. A second tornado followed about 30 minutes later outside Hyattsville, Maryland -- about 10 miles northeast of the capital. The high-speed winds, peaking at 100 mph, hit the George E. Peters Adventist School especially hard, tearing off a portion of the roof and flinging it and mounds of debris into the parking lot. A nearby construction trailer was also knocked over. E-mail to a friend.\n\nSummary:\n",
            "max_tokens": 96,
        }

        # Send the POST request to the endpoint
        start_time = time.time()
        with self.client.post("/", json=payload, catch_response=True) as response:
            response_time = time.time() - start_time
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Record data in CSV
            with open(self.csv_filename, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)

                if response.status_code == 200:
                    # Request was successful
                    status = "success"
                    response_data = response.json()
                    print(f"Success - Model: {model}, Prompt: '{prompt[:30]}...', Response time: {response_time:.2f}s")
                else:
                    # Request failed
                    status = f"failed_{response.status_code}"
                    response.failure(f"Request failed with status code {response.status_code}: {response.text}")
                    print(f"Failed - Model: {model}, Prompt: '{prompt[:30]}...', Status: {response.status_code}")

                # Write the data row
                # Include response text in the CSV output (truncated to 50 chars)
                response_text = response_data if status == "success" else ""
                writer.writerow([timestamp, model, prompt[:50], status, f"{response_time:.4f}", response_text])