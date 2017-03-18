# runWord2vec
- Gensim 라이브러리를 활용한 word2vec 훈련 및 시각화 스크립트입니다.
- This is a wrapper of Gensim word2vec along with T-SNE visualization.  
</br>

## Requirements
- Before running runWord2vec, make sure Gensim python library is installed.
- Gensim can be easily installed by:  

		$ pip install gensim
</br>

## Data preparation
- What to prepare:
	- A **text file** which has one sentence per line
	- NB. To train a set of quality word embeddings, your corpus needs to be sufficiently large.
</br>

## Functionality
- What can be done:
	- Given a text file (a corpus which has one sentence per line) in the same directory as the script, you can train your own word embeddings using the following scripts which Gensim library.   
</br>

## Usage
### 1) **runWord2vec.py**  
- Train & save word2vec model by the following command:  
	
		$ python runWord2vec.py <corpus_name> <model_name>
	
- For example:
	
		$ python runWord2vec.py wiki.txt mdl_wiki
	

### 2) **word2vecTSNE.py**
	
- Visualize your trained model by the following command:  
	
		$ python word2vecTSNE.py <model_name>
	
- For example:
	
		$ python word2vecTSNE.py mdl_wiki
</br>