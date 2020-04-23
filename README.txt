# chatbot-pytorch
	Ini merupakan chatbot yang dibuat menggunakan pytorch library 

## Run
1. Unduh terlebih dahulu datanya [disini](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)
2. Extract data yang telah diunduh
3. Ubah dalam Format.py, TrainData.py, dan Bot.py
```
	corpus = '<Nama Folder Data>'
```
4. Jalankan sintaks dibawah ini untuk membuat file format percakapan
```
    $ python Format.py
```
5. Lakukan training data 
```
    $ python TrainData.py
```
6. Jalankan botnya
```
	$ python Bot.py
```

## Documentation
	Dokumentasi ini merupakan opsi jika ingin mengubah checkpoint dan menyimpan checkpoint

- TrainData.py
```
	$ python TrainData.py --help
```
- Bot.py
```
	$ python Bot.py --help
```

## Directory Structure
```bash
├── data
│   └── *.txt
└── model
   	└── save
   		└── cb_model
			└── data
```

## Requirement
- [Python](https://www.python.org/)
- [PyTorch](https://pytorch.org/)

## Source
- [PyTorch Chatbot](https://pytorch.org/tutorials/beginner/chatbot_tutorial.html)
