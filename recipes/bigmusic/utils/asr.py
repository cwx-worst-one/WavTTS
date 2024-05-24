import requests
import time

def run_asr_lyrics_sa_online(filepath, language='zh-CN'):
    base_url = 'http://speech.byted.org/api/v1/vc'
    appid = 'bkseig7309i0'
    token = 'lv_token'
    access_token = 'YLp0eHdvLZH_IvgNSEHQrHsBeTdliqe0'
    caption_type = "singing"
    max_retry_num = 5
    try_num = 0

    text = ''
    while try_num < max_retry_num:
        try:
            with open(filepath, 'rb') as fp:
                data = fp.read()
                response = requests.post(
                             '{base_url}/submit'.format(base_url=base_url),
                             params=dict(
                                 appid=appid,
                                 token=token,
                                 language=language,
                                 caption_type=caption_type,
                                 use_itn='False',
                                 dirt_filter='False',
                                 use_capitalize='False',
                                 use_spell_correct='False',
                                 with_gender_info='False',
                                 with_speaker_info='False',
                                 max_lines=1,
                                 words_per_line=15,
                                 with_confidence='True',
                                 verbose='True'
                             ),
                             data=data,
                             headers={
                                'content-type': 'audio/m4a',
                                "Authorization": 'Bearer; {access_token}'.format(access_token=access_token)
                             }
                         )
                job_id = response.json()['id']
                response = requests.get(
                        '{base_url}/query'.format(base_url=base_url),
                        params=dict(
                            appid=appid,
                            token=token,
                            id=job_id,
                        ),
                        headers={
                            "Authorization": 'Bearer; {access_token}'.format(access_token=access_token)
                        }
                )
                for item in response.json()['utterances']:
                    text += item['text'] + '\n'
                try_num = max_retry_num + 1
        except Exception as e:
            try_num += 1
            text = ''
            time.sleep(30)
    return text.strip()



import csv

# Function to load CSV file into a list of dictionaries
def load_csv_as_dict_list(file_path):
    data = []
    with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            data.append(row)
    return data

filepath = 'VC_PGC.csv'
output_filepath = 'VC_PGC_ASR.csv'

with open(output_filepath, mode='w', newline='', encoding='utf-8') as csvfile:
    csvwriter = csv.writer(csvfile)    
    csv_data = load_csv_as_dict_list(filepath)
    for row in csv_data:
        audio_filepath = row['original_audio']
        audio_asr = run_asr_lyrics_sa_online(audio_filepath, language='zh-CN')
        # Write each pair to the CSV file        
        csvwriter.writerow((audio_filepath, audio_asr))

