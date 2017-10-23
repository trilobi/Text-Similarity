import sys
import pickle
import re
import numpy as np
import sklearn
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic

class	Match():
	def __init__(self):
		vec = []
		self.Word2Vec = pickle.load(open("../W2V/glove-300.pkl", "r"))		
	
	def process(self, sen):
		sen = re.sub("'re", " are", sen)
		sen = re.sub("n's", " is not", sen)
		sen = re.sub("'s", "", sen)
		sen = re.sub("n't", " not", sen)
		sen = re.sub("'t", "", sen)
		sen = re.sub("n'd", " not", sen)
		sen = re.sub("'d", "", sen)
		sen = re.sub("/^[a-zA-Z0-9 ]+$/", "", sen)
		return sen

	def del_stop_words(self, sen):
		stop_words = ["I", "it", "is", "was", "for", "of", "me", "you", "were", "are", "do", "did", "have", "has", "can", "could", "to", "they", "them", "and", "a", "an", "the", "be", "been"]
		sen = [i for i in sen if i not in stop_words]
		return sen

	def harmonic(self, vec1, vec2):
		try:
			mean1 = sum(vec1)/float(len(vec1))
			mean2 = sum(vec2)/float(len(vec2))
			return 1/(1/mean1+1/mean2)
		except:
			return 0
	
	def Cos_dis(self, vec1, vec2):
		vec1 = np.array(vec1)
		vec2 = np.array(vec2)
		try:
			down = np.sqrt(sum(np.square(vec1))) + np.sqrt(sum(np.square(vec2)))
			up = vec1.dot(vec2)
			sim =  up / down
			return sim
		except:
			return 0
	
	def edit_dis(self, sen1, sen2):
		list_sen1 = sen1.strip().split(" ")
		list_sen2 = sen2.strip().split(" ")
		list_sen1 = self.del_stop_words(list_sen1)
		list_sen2 = self.del_stop_words(list_sen2)
		dp = [[0 for j in range(len(list_sen2)+1)] for i in range(len(list_sen1)+1)]
		for i in range(len(list_sen1)+1):
			dp[i][0] = i
		for j in range(len(list_sen2)+1):
			dp[0][j] = j
		for i in range(len(list_sen1)):
			for j in range(len(list_sen2)):
				if list_sen1[i] == list_sen2[j]:
					dp[i+1][j+1] = dp[i][j]
				else:
					dp[i+1][j+1] = min( dp[i][j+1]+1, dp[i+1][j]+1, dp[i][j]+1)
		return dp[len(list_sen1)][len(list_sen2)]   

	def share_word(self, sen1, sen2):
		list_sen1 = sen1.strip().split(" ")
		list_sen2 = sen2.strip().split(" ")
		list_sen1 = self.del_stop_words(list_sen1)
		list_sen2 = self.del_stop_words(list_sen2)
		share = 0
		s_sen1 = [w for w in list_sen1 if w in list_sen2]
		s_sen2 = [w for w in list_sen2 if w in list_sen1]
		try:
			share = (len(s_sen1)+len(s_sen2))/(len(list_sen1)+len(list_sen2))
		except:
			pass
		return share
				
	def path_dis(self, sen1, sen2):
		list_sen1 = sen1.strip().split(" ")
		list_sen2 = sen2.strip().split(" ")
		list_sen1 = self.del_stop_words(list_sen1)
		list_sen2 = self.del_stop_words(list_sen2)
		vec_sen1 = []
		vec_sen2 = []
		for w1 in list_sen1:
			try:
				w_1 = wn.synsets(w1)[0]
			except:
				continue
			max_sim = 0
			for w2 in list_sen2:
				try:
					w_2 = wn.synsets(w2)[0]
				except:
					continue
				try:
					sim = w_1.path_similarity(w_2)
				except:
					sim = 0
				max_sim = max(max_sim, sim)
			vec_sen1.append(max_sim)
		for w1 in list_sen2:
			try:
				w_1 = wn.synsets(w1)[0]
			except:
				continue
			max_sim = 0
			for w2 in list_sen1:
				try:
					w_2 = wn.synsets(w2)[0]
				except:
					continue
				try:
					sim = w_1.path_similarity(w_2)
				except:
					sim = 0
				max_sim = max(max_sim, sim)
			vec_sen2.append(max_sim)
		return self.harmonic(vec_sen1, vec_sen2)
	
	def lch_dis(self, sen1, sen2):
		list_sen1 = sen1.strip().split(" ")
		list_sen2 = sen2.strip().split(" ")
		list_sen1 = self.del_stop_words(list_sen1)
		list_sen2 = self.del_stop_words(list_sen2)
		vec_sen1 = []
		vec_sen2 = []
		for w1 in list_sen1:
			try:
				w_1 = wn.synsets(w1)[0]
			except:
				continue
			max_sim = 0
			for w2 in list_sen2:
				try:
					w_2 = wn.synsets(w2)[0]
				except:
					continue
				try:
					sim = w_1.lch_similarity(w_2)
				except:
					sim = 0
				max_sim = max(max_sim, sim)
			vec_sen1.append(max_sim)
		for w1 in list_sen2:
			try:
				w_1 = wn.synsets(w1)[0]
			except:
				continue
			max_sim = 0
			for w2 in list_sen1:
				try:
					w_2 = wn.synsets(w2)[0]
				except:
					continue
				try:
					sim = w_1.lch_similarity(w_2)
				except:
					sim = 0
				max_sim = max(max_sim, sim)
			vec_sen2.append(max_sim)
		return self.harmonic(vec_sen1, vec_sen2)
	
	def wup_dis(self, sen1, sen2):
		list_sen1 = sen1.strip().split(" ")
		list_sen2 = sen2.strip().split(" ")
		list_sen1 = self.del_stop_words(list_sen1)
		list_sen2 = self.del_stop_words(list_sen2)
		vec_sen1 = []
		vec_sen2 = []
		for w1 in list_sen1:
			try:
				w_1 = wn.synsets(w1)[0]
			except:
				continue
			max_sim = 0
			for w2 in list_sen2:
				try:
					w_2 = wn.synsets(w2)[0]
				except:
					continue
				try:
					sim = w_1.wup_similarity(w_2)
				except:
					sim = 0
				max_sim = max(max_sim, sim)
			vec_sen1.append(max_sim)
		for w1 in list_sen2:
			try:
				w_1 = wn.synsets(w1)[0]
			except:
				continue
			max_sim = 0
			for w2 in list_sen1:
				try:
					w_2 = wn.synsets(w2)[0]
				except:
					continue
				try:
					sim = w_1.wup_similarity(w_2)
				except:
					sim = 0
				max_sim = max(max_sim, sim)
			vec_sen2.append(max_sim)
		return self.harmonic(vec_sen1, vec_sen2)
	
	def jcn_dis(self, sen1, sen2):
		list_sen1 = sen1.strip().split(" ")
		list_sen2 = sen2.strip().split(" ")
		list_sen1 = self.del_stop_words(list_sen1)
		list_sen2 = self.del_stop_words(list_sen2)
		vec_sen1 = []
		vec_sen2 = []
		brown_ic = wordnet_ic.ic("ic-brown.dat")
		for w1 in list_sen1:
			try:
				w_1 = wn.synsets(w1)[0]
			except:
				continue
			max_sim = 0
			for w2 in list_sen2:
				try:
					w_2 = wn.synsets(w2)[0]
				except:
					continue
				try:
					sim = w_1.jcn_similarity(w_2, brown_ic)
				except:
					sim = 0
				max_sim = max(max_sim, sim)
			vec_sen1.append(max_sim)
		for w1 in list_sen2:
			try:
				w_1 = wn.synsets(w1)[0]
			except:
				continue
			max_sim = 0
			for w2 in list_sen1:
				try:
					w_2 = wn.synsets(w2)[0]
				except:
					continue
				try:
					sim = w_1.jcn_similarity(w_2, brown_ic)
				except:
					sim = 0 
				max_sim = max(max_sim, sim)
			vec_sen2.append(max_sim)
		return min(10, self.harmonic(vec_sen1, vec_sen2))
	
	def res_dis(self, sen1, sen2):
		list_sen1 = sen1.strip().split(" ")
		list_sen2 = sen2.strip().split(" ")
		list_sen1 = self.del_stop_words(list_sen1)
		list_sen2 = self.del_stop_words(list_sen2)
		vec_sen1 = []
		vec_sen2 = []
		brown_ic = wordnet_ic.ic("ic-brown.dat")
		for w1 in list_sen1:
			try:
				w_1 = wn.synsets(w1)[0]
			except:
				continue
			max_sim = 0
			for w2 in list_sen2:
				try:
					w_2 = wn.synsets(w2)[0]
				except:
					continue
				try:
					sim = w_1.res_similarity(w_2, brown_ic)
				except:
					sim = 0
				max_sim = max(max_sim, sim)
			vec_sen1.append(max_sim)
		for w1 in list_sen2:
			try:
				w_1 = wn.synsets(w1)[0]
			except:
				continue
			max_sim = 0
			for w2 in list_sen1:
				try:
					w_2 = wn.synsets(w2)[0]
				except:
					continue
				try:
					sim = w_1.res_similarity(w_2, brown_ic)
				except:
					sim = 0 
				max_sim = max(max_sim, sim)
			vec_sen2.append(max_sim)
		return min(10, self.harmonic(vec_sen1, vec_sen2))
	
	def lin_dis(self, sen1, sen2):
		list_sen1 = sen1.strip().split(" ")
		list_sen2 = sen2.strip().split(" ")
		list_sen1 = self.del_stop_words(list_sen1)
		list_sen2 = self.del_stop_words(list_sen2)
		brown_ic = wordnet_ic.ic("ic-brown.dat")
		vec_sen1 = []
		vec_sen2 = []
		for w1 in list_sen1:
			try:
				w_1 = wn.synsets(w1)[0]
			except:
				continue
			max_sim = 0
			for w2 in list_sen2:
				try:
					w_2 = wn.synsets(w2)[0]
				except:
					continue
				try:
					sim = w_1.lin_similarity(w_2, brown_ic)
				except:
					sim = 0
				max_sim = max(max_sim, sim)
			vec_sen1.append(max_sim)
		for w1 in list_sen2:
			try:
				w_1 = wn.synsets(w1)[0]
			except:
				continue
			max_sim = 0
			for w2 in list_sen1:
				try:
					w_2 = wn.synsets(w2)[0]
				except:
					continue
				try:
					sim = w_1.lin_similarity(w_2, brown_ic)
				except:
					sim = 0
				max_sim = max(max_sim, sim)
			vec_sen2.append(max_sim)
		return self.harmonic(vec_sen1, vec_sen2)
	
	def W2V_sim(self, sen1, sen2):
		list_sen1 = sen1.strip().split(" ")
		list_sen2 = sen2.strip().split(" ")
		list_sen1 = self.del_stop_words(list_sen1)
		list_sen2 = self.del_stop_words(list_sen2)
		sen1_vec = []
		sen2_vec = []
		for w1 in list_sen1:
			 if self.Word2Vec.has_key(w1):
				sen1_vec.append(self.Word2Vec[w1])
		for w2 in list_sen2:
			if self.Word2Vec.has_key(w2):
				sen2_vec.append(self.Word2Vec[w2])
		if len(sen1_vec)!=0 and len(sen2_vec)!=0:
			sen1_vec = np.array(sen1_vec)
			sen2_vec = np.array(sen2_vec)
			sen1_mean = sen1_vec.mean(0)
			sen2_mean = sen2_vec.mean(0)
			return self.Cos_dis(sen1_mean, sen2_mean)
		else:
			return 0	
	
	def Sen2Vec(self, sen1, sen2):
		list_sen1 = sen1.strip().split(" ")
		list_sen2 = sen2.strip().split(" ")
		list_sen1 = self.del_stop_words(list_sen1)
		list_sen2 = self.del_stop_words(list_sen2)
		vec1 = []
		vec2 = []
		reesult = 0
		for w1 in list_sen1:
			if self.Word2Vec.has_key(w1):
				w1_vec = self.Word2Vec[w1]
				max_sim = 0
				for w2 in list_sen2:
					if self.Word2Vec.has_key(w2):
						w2_vec = self.Word2Vec[w2]
						max_sim = max(max_sim, self.Cos_dis(w1_vec, w2_vec))
				vec1.append(max_sim)
		for w1 in list_sen2:
			if self.Word2Vec.has_key(w1):
				w1_vec = self.Word2Vec[w1]
				max_sim = 0
				for w2 in list_sen1:
					if self.Word2Vec.has_key(w2):
						w2_vec = self.Word2Vec[w2]
						max_sim = max(max_sim, self.Cos_dis(w1_vec, w2_vec))
				vec2.append(max_sim)
		return self.harmonic(vec1, vec2)
	
	def Avg_Vec(self, sen1, sen2):
		list_sen1 = sen1.strip().split(" ")
		list_sen2 = sen2.strip().split(" ")
		list_sen1 = self.del_stop_words(list_sen1)
		list_sen2 = self.del_stop_words(list_sen2)
		vec = []
		for w1 in list_sen1:
			if self.Word2Vec.has_key(w1):
				w1_vec = self.Word2Vec[w1]
				max_sim = 0
				for w2 in list_sen2:
					if self.Word2Vec.has_key(w2):
						w2_vec = self.Word2Vec[w2]
						max_sim = max(max_sim, self.Cos_dis(w1_vec, w2_vec))
				vec.append(max_sim)
		vec = np.array(vec)
		mean = np.mean(vec)
		return mean

	def Media_Vec(self, sen1, sen2):
		list_sen1 = sen1.strip().split(" ")
		list_sen2 = sen2.strip().split(" ")
		list_sen1 = self.del_stop_words(list_sen1)
		list_sen2 = self.del_stop_words(list_sen2)
		vec = []
		for w1 in list_sen1:
			if self.Word2Vec.has_key(w1):
				w1_vec = self.Word2Vec[w1]
				max_sim = 0
				for w2 in list_sen2:
					if self.Word2Vec.has_key(w2):
						w2_vec = self.Word2Vec[w2]
						max_sim = max(max_sim, self.Cos_dis(w1_vec, w2_vec))
				vec.append(max_sim)
		vec = np.array(vec)
		vec = np.msort(vec)
		media = np.median(vec)
		return media

	def Var_Vec(self, sen1, sen2):
		list_sen1 = sen1.strip().split(" ")
		list_sen2 = sen2.strip().split(" ")
		list_sen1 = self.del_stop_words(list_sen1)
		list_sen2 = self.del_stop_words(list_sen2)
		vec = []
		for w1 in list_sen1:
			if self.Word2Vec.has_key(w1):
				w1_vec = self.Word2Vec[w1]
				max_sim = 0
				for w2 in list_sen2:
					if self.Word2Vec.has_key(w2):
						w2_vec = self.Word2Vec[w2]
						max_sim = max(max_sim, self.Cos_dis(w1_vec, w2_vec))
				vec.append(max_sim)
		vec = np.array(vec)
		var = np.var(vec)
		return var

	def neg_word(self, sen1, sen2):
		neg = ["cannot", "rather", "no", "not", "hardly","but", "barely", "seldom", "rarely", "little", "few", "nothing", "never", "without"]
		list_sen1 = sen1.strip().split(" ")
		list_sen2 = sen2.strip().split(" ")
		list_sen1 = self.del_stop_words(list_sen1)
		list_sen2 = self.del_stop_words(list_sen2)
		neg_1 = 0
		neg_2 = 0
		if neg in list_sen1:
			neg_1 += 1
		if neg in list_sen2:
			neg_2 += 1
		return (neg_1%2) ^ (neg_2%2)	
	
	def get_feature(self, sen1, sen2):
		vec = []
		vec.append(self.edit_dis(sen1, sen2))
		vec.append(self.share_word(sen1, sen2))
		vec.append(self.path_dis(sen1, sen2))
		vec.append(self.lch_dis(sen1, sen2))
		vec.append(self.wup_dis(sen1, sen2))
		vec.append(self.jcn_dis(sen1, sen2))
		vec.append(self.res_dis(sen1, sen2))
		vec.append(self.lin_dis(sen1, sen2))
		vec.append(self.W2V_sim(sen1, sen2))
		vec.append(self.Sen2Vec(sen1, sen2))
		vec.append(self.neg_word(sen1, sen2))
		vec.append(self.Avg_Vec(sen1, sen2))
		vec.append(self.Media_Vec(sen1, sen2))
		vec.append(self.Var_Vec(sen1, sen2))
		return vec
				
	
if __name__ == "__main__":
	'''
	sen1 = "I die a little but still do not mind bad tv"
	sen2 = "Reality shows cause me to die a little whenever I see them advertised even though I have watched them at times."
	tool = Match()
	sen1 = tool.process(sen1)
	sen2 = tool.process(sen2)
	print tool.share_word(sen1, sen2)
	print tool.Avg_Vec(sen1, sen2)
	print tool.Media_Vec(sen1, sen2)
 	print tool.Var_Vec(sen1, sen2)
	'''
	train_file = open("../data/row/train-full.txt")
	dev_file = open("../data/row/dev-full.txt")
	train_file.readline()
	dev_file.readline()
	train_vec = []
	train_labels = []
	dev_vec = []
	dev_labels = []
	tool = Match()
	for line in train_file.readlines():
		vec = []
		items = line.strip().split("\t")
		if len(items) < 8:
			continue
		sen1 = tool.process(items[1])
		sen2 = tool.process(items[4])
		train_labels.append(1-int(items[3]))
	 	train_vec.append(tool.get_feature(sen1, sen2))
		sen1 = tool.process(items[2])
		sen2 = tool.process(items[4])
		train_vec.append(tool.get_feature(sen1, sen2))	
		train_labels.append(int(items[3]))

	for line in dev_file.readlines():
		items = line.strip().split("\t")
		if len(items) < 8:
			continue
		sen1 = tool.process(items[1])
		sen2 = tool.process(items[4])
		dev_vec.append(tool.get_feature(sen1, sen2))
		sen1 = tool.process(items[2])
		sen2 = tool.process(items[4])
		dev_vec.append(tool.get_feature(sen1, sen2))
		dev_labels.append(int(items[3]))
	np.save("../data/feature/train_vec.npy", np.array(train_vec))
	np.save("../data/feature/train_labels.npy", np.array(train_labels))
	np.save("../data/feature/dev_vec.npy", np.array(dev_vec))
	np.save("../data/feature/dev_lables.npy", np.array(dev_labels))
	
