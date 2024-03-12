
import librosa
import torch
import torch_museval
import os


def get_wav_from_path(path):
    wav_dict = {}
    file_list = os.listdir(path)
    for filename in file_list:
        name, ext = os.path.splitext(filename)
        if ext == '.wav':
            wav_dict[name] = os.path.join(path, filename)
    return wav_dict



class Measures():
    def __init__(
        self,
        ref_path, 
        rec_path,
        comment = "",
    ):
        self.ref_path = ref_path
        self.rec_path = rec_path
        self.statistic = {}
        if comment == "":
            self.comment = os.path.basename(os.path.abspath(rec_path))
        else:
            self.comment = comment

    
    def __repr__(self):
        repr = f"Measures of {self.comment} num = {len(self.measures_dict)}"
        for key in self.statistic:
            repr += "\n\t{} {:5.5}|{:5.5}".format(key, self.statistic[key].mean(), self.statistic[key].var())
        return repr
    

    def get_measures_pair(
        self,
        ref_wav_path, 
        rec_wav_path, 
        sr = 24000,
        win = 24000*10,
        hop = 24000):

        ref_wav, _ = librosa.load(ref_wav_path, sr=sr, mono=True)
        rec_wav, _ = librosa.load(rec_wav_path, sr=sr, mono=True)

        ref_wav = torch.FloatTensor(ref_wav).unsqueeze(0)
        rec_wav = torch.FloatTensor(rec_wav).unsqueeze(0)

        sdr, isr, sir, sar = torch_museval.evaluate(
            ref_wav.T.unsqueeze(0).detach(),
            rec_wav.T.unsqueeze(0).detach(),
            win=win, hop=hop,
        )

        return {
            "sdr" : sdr,
            "isr" : isr,
            "sir" : sir,
            "sar" : sar,
            }


    def run(
        self,
        sr = 24000,
        win = 24000*10,
        hop = 24000,
        ):
            ref_wav_dict = get_wav_from_path(self.ref_path)
            rec_wav_dict = get_wav_from_path(self.rec_path)

            cmp_wav_dict = {}
            for filename in rec_wav_dict:
                if filename in ref_wav_dict:
                    cmp_wav_dict[filename] = ( ref_wav_dict[filename], rec_wav_dict[filename] )
            
            self.measures_dict = {}
            for filename in cmp_wav_dict:
                measures = self.get_measures_pair(
                    *cmp_wav_dict[filename], 
                    sr = sr,
                    win = win,
                    hop = hop)
                self.measures_dict[filename] = measures
            
            self.get_statistic('sdr')
    

    def get_statistic(
        self,
        key='sdr'
        ):
        # - filename1:
        #     - sdr:
        #     - isr:
        #     - sir:
        #     - sar:
        # - filename2:
        #     - sdr:
        #     - isr:
        #     - sir:
        #     - sar:

        target_statistic = None
        for filename in self.measures_dict:
            if target_statistic == None:
                target_statistic = self.measures_dict[filename][key]
            else:
                target_statistic = torch.cat([target_statistic, self.measures_dict[filename][key]],dim=1)
        self.statistic[key] = target_statistic
    

    def compare(self, other, key='sdr'):

        cmp_measures_dict = {}
        for filename in self.measures_dict:
            assert filename in other.measures_dict
            cmp_measures_dict[filename] = (
                self.measures_dict[filename][key].mean() - other.measures_dict[filename][key].mean(),
                self.measures_dict[filename][key].mean(),
                other.measures_dict[filename][key].mean(),
            )
        
        cmp_measures_dict = sorted(cmp_measures_dict.items(), key=lambda x:x[1][0])
        cmp_measures_dict = dict(cmp_measures_dict)
        print(self)
        print(other)
        print("comparing result:")
        for filename in cmp_measures_dict:
            print("\n\t{:<50} {:5.5}|{:5.5}|{:5.5}".format(filename, 
                cmp_measures_dict[filename][0],
                cmp_measures_dict[filename][1],
                cmp_measures_dict[filename][2],
                ))






win = 24000*10
hop = 24000
ref_path = "/mnt/bn/data-storage-hl/user/zhangshuo/data/assets/mix_vocal/audio/"

measures1 = Measures(ref_path, "/mnt/bn/data-storage-hl/user/zhangshuo/data/tmp/20240310/exp1_160k/")
measures1.run()
measures2 = Measures(ref_path, "/mnt/bn/data-storage-hl/user/zhangshuo/data/tmp/20240310/exp25_260k/")
measures2.run()

measures1.compare(measures2)








