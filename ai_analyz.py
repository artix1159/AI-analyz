import requests
import json
import concurrent.futures
from openai import OpenAI
import os
import assemblyai as aai
from datetime import datetime


class ChatDataFetcher:
    def __init__(self, url, file_name):
        self.url = url
        self.file_name = file_name

    def fetch_data(self, dialog_count):
        existing_data = {'data': []}
        while len(existing_data['data']) < dialog_count:
            response = requests.get(self.url)
            if response.status_code == 200:
                data = response.json()
                new_data = self.remove_dialogs_with_non_empty_urls(data)
                existing_data['data'].extend(new_data['data'])
            else:
                print(f"Error: {response.status_code}")
                break

        existing_data['data'] = existing_data['data'][:dialog_count]
        with open(self.file_name, 'w', encoding='utf-8') as file:
            json.dump(existing_data, file, ensure_ascii=False, indent=4)
        return existing_data

    @staticmethod
    def remove_dialogs_with_non_empty_urls(data):
        return {
            'data': [
                item for item in data['data']
                if not any(message['url'] for message in item['messages'])
            ]
        }


class AudioTranscriber:
    def __init__(self, directory_path, language_code, api_key):
        self.data = None
        self.directory_path = directory_path
        self.language_code = language_code
        self.api_key = api_key
        aai.settings.api_key = self.api_key  # Встановлюємо API ключ для AssemblyAI
        self.transcriber = aai.Transcriber()
        self.transcripts = []

    def transcribe_file(self, file_path):
        config = aai.TranscriptionConfig(dual_channel=True, language_code=self.language_code)
        transcript = self.transcriber.transcribe(file_path, config=config)
        return {
            "id": transcript.id,
            "messages": [
                {
                    "speaker": utterance.speaker,
                    "message": utterance.text,
                    "start": utterance.start,
                    "end": utterance.end
                } for utterance in transcript.utterances
            ]
        }

    def transcribe(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            file_paths = [os.path.join(self.directory_path, filename) for filename in os.listdir(self.directory_path) if
                          filename.endswith('.wav')]
            futures = [executor.submit(self.transcribe_file, file_path) for file_path in file_paths]
            for future in futures:
                self.transcripts.append(future.result())
        self.data = {"data": self.transcripts}
        self.save_transcripts('transcripts.json')

    def save_transcripts(self, output_file):
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=4)


class GPTDataProcessor:
    def __init__(self, api_key, input_file_name, output_file_name):
        self.client = OpenAI(api_key=api_key)
        self.input_file_name = input_file_name
        self.output_file_name = output_file_name

    def process_data(self):
        with open(self.input_file_name, 'r', encoding='utf-8') as file:
            existing_data = json.load(file)

        gpt_responses = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            future_to_dialog = {executor.submit(self.process_dialog, dialog): dialog for dialog in
                                existing_data['data']}
            for future in concurrent.futures.as_completed(future_to_dialog):
                dialog = future_to_dialog[future]
                try:
                    gpt_response = future.result()
                    gpt_responses.append(gpt_response)
                except Exception as e:
                    print(f"An error occurred during dialog processing: {e}")

        with open(self.output_file_name, 'w', encoding='utf-8') as file:
            json.dump(gpt_responses, file, ensure_ascii=False, indent=4)

    def process_dialog(self, dialog):
        messages = [
            {"role": "system", "content": """

    Your task is to analyze a given dialog and provide an answer in the form of a JSON object containing the following properties:

    1. "dialogue_quality_score": A score from 0 to 100. It depends on all of the following elements:
    Assess whether the operator maintained a consistent tone of professionalism regardless of the customer's behavior.
    Rate the level of empathy and understanding demonstrated by the operator in response to the customer's situation.
    Assess how clearly the operator communicated information and instructions.
    Evaluate how effectively the operator identified and resolved the customer's underlying problem
    Evaluate the operator's level of engagement during the call. Did he or she respond quickly and appropriately to the customer's requests?
    Assess how well the agent adapted his/her approach depending on the course of the conversation or the emotional state of the customer
    Evaluate how effectively the client's concerns were addressed.
    Evaluate how personalized the agent's interaction with the client was to meet the client's specific needs or circumstances.
    Considering all aspects of the conversation, assess the overall satisfaction of the customer
    2. "dialog_theme": The main topic discussed during the dialog. Only one of the topics presented needs to be selected: 
    Select "Фінанси" if the main topic of dialog is related to:
    Payments.
    Props.
    Problems with payment that did not arise from the payment system.
    Payment deadlines.
    Requests not to disconnect
    provide credit for recharging.
    To enable a service.
    Select "Обслуговування" if the main topic of dialog is related to:
    Servicing the subscriber for Triolan issues that do not affect the operation of the services.
    Updating contact information (correcting phone number, name, reissuing a contract). 
    Package change (example: changing tariff from 1GB to 100MB and back). 
    Choose "Відключення" if the main topic of the dialog is related to:
    Disconnection with and without application fulfillment/completion. 
    Choose "Ремонт" if the main topic of the dialog is related to:
    Causes of poor service delivery, in the absence of a fixed accident with or without execution/completion of the request. 
    Select "Повторна активація" if the main topic of the dialog is related to:
    With the creation of a Reactivation(PA) application.
    The cost and terms of Reactivation(PA) (including reasons for not being able to recharge a deactivated Reactivation(PA)).
    Calls from suspected Nissan (unauthorized connections) of KTV subscribers.
    The presence/absence of promotions or discounts on Reactivation(PA). 
    When calling the wizard for reactivation.
    Select "Підключення-Нове" if questions arise in the call:
    On availability and connection with or without application; 
    Checking the technical feasibility of connection 
    Presence/absence of promotions or discounts for new connections.  
    Select "Аварія" if the main topic of the dialog is related to:
    Reasons for poor service quality, if an accident is recorded for this service (with or without deadlines).
    Select "Незрозумілий звінок" if the main topic of dialog is related to:
    Incomprehensible caller questions.
    Problems with communication. 
    Specify the topic of the dialog in Ukrainian.
    3. "filler_words": words that do not carry a semantic load and are often repeated.
    4. "obscene_lexicon": foul language used in the dialog.
    5. "keywords": A list of words that are important to the dialog.
    6. "client_mood_analysis": A description of the client's mood throughout the dialog.
    7. "operator_mood_analysis": Description of the operator's mood throughout the dialog.
    8. "key_moments": A list of significant or notable events or exchanges that occurred during the dialog.
    9. "operator_errors": a list of operator errors:
    not saying hello, being rude to a customer, not solving a customer's problem, not asking at the end of the conversation "are there any more questions", etc.

    you are obliged to provide a response in json format
    Response in Ukrainian
            """},
            {"role": "user", "content": ""}
        ]
        for message in dialog['messages']:
            messages.append({"role": "user", "content": message['message']})

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=messages,
            temperature=0,
            max_tokens=2000,
            top_p=0,
            frequency_penalty=0,
            presence_penalty=0
        )

        answer = response.choices[0].message.content.replace("\n", "")
        if "```json" in answer:
            answer = answer.replace("```json", "").replace("```", "")
        return {
            'dialog_id': dialog['id'],
            'gpt_response': json.loads(answer)
        }


url = "API for chats"
file_name = 'chats.json'
#
start_time = datetime.now()

fetcher = ChatDataFetcher(url, file_name)
fetcher.fetch_data(dialog_count=100)

api_key = "OpenAI API"
input_file_name = 'new_file.json'
output_file_name = 'output_GPT_chat.json'

processor = GPTDataProcessor(api_key, input_file_name, output_file_name)

processor.process_data()

# DIRECTORY_PATH = r"D:\coding\python\pycharm\ai_analyzer\calls"
# LANGUAGE_CODE = "uk"
# API_KEY = "AssemblyAI API"
#
# transcriber = AudioTranscriber(DIRECTORY_PATH, LANGUAGE_CODE, API_KEY)
# transcriber.transcribe()
#
# api_key = "OpenAI API"
# input_file_name = 'transcripts.json'
# output_file_name = 'output_GPT_dialog.json'
#
# processor = GPTDataProcessor(api_key, input_file_name, output_file_name)
#
# processor.process_data()

end_time = datetime.now()
print(f"execution time: {end_time - start_time}")
