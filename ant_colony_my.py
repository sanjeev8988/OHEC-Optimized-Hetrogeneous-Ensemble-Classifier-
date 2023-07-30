#import Thread

class ant_colony:			#class ant_colony to create ant objects and pheromone updation and all
	
	class ant():			# class to create ant
		def __init__(self, init_classifier, possible_classifier,list_of_classifier, pheromone_list, pred_dict,y_test,eta, first_pass= False):
			#Thread.__init__(self)
			''' initializing the parameter
			init_classifier-> (list)initial set of classifier 
			possible_classifier-> (list) list of other possible classifier, which can be added
			list_of_classifier-> (dictionary)list of all classifier
			pheromone_list-> (dictionary) contain pheromone value for all classifier
			pred_dict-> (dictionary) contain the perdicted value for each classifier
			y_test-> (list)actual values to testing 
			first_pass-> flag to check first_pass or not
			'''
			self.init_classifier=init_classifier
			self.possible_classifier=possible_classifier
			self.pheromone_list=pheromone_list
			self.list_of_classifier=list_of_classifier
			self.pred_dict=pred_dict
			self.current_classifiers=init_classifier
			self.first_pass=first_pass
			self.pqe=0.0;
			self.y_test=y_test
			self.Q=1
			self.delta_tau={}
			self.eta=eta
			
		def calculate_PQE(self,pred_dict,list_of_classifier,classifiers):
			'''calculate PQE for best classifier selection'''
			import math
			l=len(classifiers)
			div=[]
			acc=[]
			for x in range(0,l):
				for y in range(x+1,l):
					q,j=self.get_div_acc(self.y_test,pred_dict[classifiers[x]],pred_dict[classifiers[y]])
					div.append(q)
					acc.append(j)
					
				total_div=2*(sum(div))/(l*(l-1))
				total_acc=2*(sum(acc))/(l*(l-1))

			U_temp=l*total_acc
			L_temp=math.sqrt(l+(l*(l-1))*(1-total_div))
			merit_PQE=U_temp/L_temp
			self.pqe=merit_PQE
			return(merit_PQE)  
		
		def get_div_acc(self,y_test,pred_dict1,pred_dict2):
			''' calculate diversity and accuracy for PQE calculation'''
			import math
			N_11=N_10=N_01=N_00=0
			for x in range(0,len(y_test)):
				if (pred_dict1[x]==y_test[x])&(pred_dict2[x]==y_test[x]):
					N_11=N_11+1
				if (pred_dict1[x]==y_test[x])&(pred_dict2[x]!=y_test[x]):
					N_10=N_10+1
				if (pred_dict1[x]!=y_test[x])&(pred_dict2[x]==y_test[x]):
					N_01=N_01+1
				if (pred_dict1[x]!=y_test[x])&(pred_dict2[x]!=y_test[x]):
					N_00=N_00+1
			N=N_11+N_10+N_01+N_00
			a=N_11/N
			#print('a=',a)
			b=N_10/N
			#print('b=',b)
			c=N_01/N
			#print('c=',c)
			d=N_00/N
			#print('d=',d)
			#print('sum=',a+b+c+d)
			acc=a/(a+b+c+d)
			#j_acc_l.append(acc)
			q_stat=(a*d-b*c)/(a*d+b*c)
			#q_stat_l.append(q_stat)
			dis_msr=b+c
			#dis_msr_l.append(dis_msr)
			df_msr= d
			#df_msr_l.append(df_msr)
			M1=a*d-b*c
			M2=math.sqrt((a+b)*(c+d)*(a+c)*(b+d))
			corr_coef=M1/M2
			#corr_coef_l.append(corr_coef)
			return corr_coef,acc
		
		def get_PQE(self):
			'''return PQE value'''
			pqe=self.calculate_PQE(self.pred_dict,self.list_of_classifier,self.current_classifiers)
			return pqe
			
		def run(self):
			"""start the execution, if first_pass calculate the PQE with diversity and accuracy and return else pick_classifier() 
			and then calculate the PQE for that classifier"""
			if self.first_pass:
				PQE=self.calculate_PQE(self.pred_dict,self.list_of_classifier,self.init_classifier)
			else:
				# add next classifier in the classifier list 
				for x in self.possible_classifier:
					cl= self.pick_classifier()
					temp=self.decision_pick_or_not(cl)
					if temp==1:
						self.update_list(cl)
					PQE=self.calculate_PQE(self.pred_dict,self.list_of_classifier,self.current_classifiers)
			print("current_classifiers: ", self.current_classifiers)
			print("PQE: ",PQE)
			self.calculate_delta_tau()
			return PQE
			
		def decision_pick_or_not(self, new_classifier):
			pqe_old=self.calculate_PQE(self.pred_dict,self.list_of_classifier,self.current_classifiers)
			current_classifier2=[i for i in self.current_classifiers]
			current_classifier2.append(new_classifier)
			pqe_new=self.calculate_PQE(self.pred_dict,self.list_of_classifier,current_classifier2)
			if(pqe_new>pqe_old):
				return 1
			else:
				return 0
		
		def pick_classifier(self):
			"""calculate the probability for selecting a classifier from possible_classifier list, based on their maximum probability of a classifier"""
			import math
			sum=0
			phero={}
			alpha=0.5
			beta=0.5
			all_classifier=list(self.list_of_classifier.keys())
			Not_in_list=[x for x in all_classifier if x not in self.init_classifier]
			for x in self.init_classifier:
				sum=sum+((self.pheromone_list[x]**alpha)*(self.eta[x]**beta))
				
			for x in Not_in_list:
				phero[x]=((self.pheromone_list[x]**alpha)*(self.eta[x]**beta))/sum
				
			key_max = max(phero.keys(), key=(lambda k: phero[k]))
			return key_max
		
		def update_list(self, cl):
			""" update the selected list_of_classifier for that ant"""
			self.current_classifiers.append(cl)
			
		def get_classifier(self):
			return self.current_classifiers
		
		def calculate_delta_tau(self):
			for x in list(self.list_of_classifier):
				if x in self.current_classifiers:
					self.delta_tau[x]=self.Q/len(self.current_classifiers)
					#print(x,' :',delta_tau[x])
				else:
					self.delta_tau[x]=0.0
					#print(x,' :',delta_tau[x])
				
			#return delta_tau
		
		
	def __init__(self, list_of_classifier, review_set, review_sentiment, start=None, ant_count=10, pred_dict={}, evaporation_constant=.4, pheromone_constant=1000.0, iterations=10):
		""" initialize all values and """
		self.list_of_classifier=list_of_classifier
		self.first_pass=True
		self.ant_count=ant_count
		self.pred_dict=pred_dict
		self.review_set=review_set
		self.review_sentiment=review_sentiment
		self.evaporation_constant=evaporation_constant
		self.pheromone_constant=pheromone_constant
		self.iterations=iterations
		self.pheromone_list={}
		self.ants=[]
		self.y_test=[]
		self.eta={}
		
		print('ant_colony constructor')
		
		if type(list_of_classifier) is not dict:
			raise TypeError("list_of_classifier must be dict")
		
		if len(list_of_classifier) < 1:
			raise ValueError("there must be at least one classifier in dict nodes")
			
		#create_pred_dict(self.review_set,self.review_sentiment,self.pred_dict)
		
	
		
		
	def create_pred_dict(self):
		""" create a dictionary which stores predicted result for each classifier"""
		import sklearn
		from sklearn.cross_validation import train_test_split
		from sklearn.metrics import classification_report
		from sklearn import metrics
		X_train, X_test, y_train, y_test=train_test_split(self.review_set,self.review_sentiment,random_state=1)
		from sklearn.feature_extraction.text import CountVectorizer
		vect=CountVectorizer(stop_words='english',ngram_range=(1, 2),max_features=500)
		vect.fit(X_train)
		X_train_dtm=vect.transform(X_train)
		X_test_dtm=vect.transform(X_test)
		self.y_test=y_test
		
		keys_all=list(self.list_of_classifier)
		for x in keys_all:
			self.list_of_classifier[x].fit(X_train_dtm,y_train)
			self.pred_dict[x]=self.list_of_classifier[x].predict(X_test_dtm)
			print("prediction completed of : ",x)
			self.eta[x]=metrics.accuracy_score(y_test,self.pred_dict[x])
			print('Accuracy Score: ',self.eta[x])
			report= classification_report(y_test,self.pred_dict[x])
			print(report)

	def create_pheromone_list(self):
		cc=1/(len(self.list_of_classifier))
		for x in self.list_of_classifier.keys():
			self.pheromone_list[x]=cc
		
	def init_ants(self):
		import random
		all_classifier=self.list_of_classifier.keys()
		if self.first_pass:
			#print("x is true")
			print("tag_1: in first pass, for 10 ants... ")
			for z in range(0,self.ant_count):
				init_classifier=random.sample(self.list_of_classifier.keys(),3)
				print("tag_1:  ",init_classifier)
				
				possible_classifier=[x for x in all_classifier if x not in init_classifier]
				self.ants.append(self.ant(init_classifier,possible_classifier,self.list_of_classifier,self.pheromone_list,self.pred_dict,self.y_test,self.eta,first_pass=True))
		else:
			#print("x is not true")
			print("tag_2: Creating random sample for next pass... ")
			for z in range(self.ant_count):
				init_classifier=random.sample(self.list_of_classifier.keys(),3)
				
				possible_classifier=[x for x in all_classifier if x not in init_classifier]
				self.ants[z].__init__(init_classifier,possible_classifier,self.list_of_classifier,self.pheromone_list,self.pred_dict,self.y_test,self.eta,first_pass=False)
		for z in range(self.ant_count):
			self.ants[z].run()
			
			
		if self.first_pass:
			self.first_pass=False
		
		
	
	def _update_pheromone_list(self):
		PQE_list=[]
		top_classifiers=[]
		top=5
		""" update the pheromone for each classifier using the values of each ant"""
		for x in range(self.ant_count):
			PQE_list.append(self.ants[x].get_PQE())
		
		top_ten=sorted(range(len(PQE_list)), key=lambda i: PQE_list[i], reverse=True)[:top]
		for x in range(0,top):
			top_classifiers.append(self.ants[top_ten[x]].get_classifier())
		
		for y in range(0,top):	
			for x in top_classifiers[y]:
				self.pheromone_list[x]=(1-self.evaporation_constant)*self.pheromone_list[x]+ self.calculate_sum_delta_tau(x)
		
	def calculate_sum_delta_tau(self,x):
		sum=0
		#print('inside calculate_delta_tau:',x)
		for y in range(self.ant_count):
			sum=sum+self.ants[y].delta_tau[x]
		return sum
		


	def _populate_ant_updated_pheromone_map(self, ant):
		""" print the updated pheromone map and return the updated pheromone list"""
	def print_ensemble_classifiers(self):
		for x in range(self.ant_count):
			print(self.ants[x].get_classifier())
			print(self.ants[x].pqe)
	
	def mainloop(self):
		""" Runs the worker ants, collects their returns and updates the pheromone map with pheromone values from workers
			calls:
			_update_pheromones()
			ant.run()
		runs the simulation self.iterations times"""
		self.create_pred_dict()
		self.create_pheromone_list()
		
		for x in range(self.iterations):
			print('for iteration:',x)
			self.init_ants()
			self._update_pheromone_list()
			
		self.print_ensemble_classifiers()
		
		
		
def hello_python():
	print("hello python: the game begins now...")
	from sklearn import svm
	from sklearn.neighbors import KNeighborsClassifier
	from sklearn.neural_network import MLPClassifier									
	from sklearn.ensemble import AdaBoostClassifier
	from sklearn.linear_model import SGDClassifier
	from sklearn.naive_bayes import MultinomialNB
	from sklearn.tree import DecisionTreeClassifier
	from sklearn.linear_model import LogisticRegression
	from sklearn.ensemble import RandomForestClassifier
	rfc = RandomForestClassifier(random_state=1)
	lr = LogisticRegression(random_state=1)
	dt=DecisionTreeClassifier( criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_split=0, class_weight=None, presort=False)
	nb=MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
	sgd=SGDClassifier()
	ada= AdaBoostClassifier()
	mlp=MLPClassifier()
	knn=KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1)
	sv=svm.SVC(C=10.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None)
		
	list_of_classifier={'svm':sv, 'knn':knn, 'mlp':mlp, 'ada':ada, 'sgd': sgd,'mnb': nb, 'dt':dt,'lr':lr, 'rfc':rfc}
	#pheromone_map=create_pheromone_list(list_of_classifier)
	review_set,review_sentiment=create_dataset()
	ant_c=ant_colony(list_of_classifier,review_set,review_sentiment)
	ant_c.mainloop()
	
	
def create_dataset():
	import json
	reviews = []
	for line in open('Cell_Phones_and_Accessories_5.json', 'r'):
		reviews.append(json.loads(line))

	review_text=[]
	review_star=[]

	for l in range(0,len(reviews)):
		review_text.append(reviews[l]['reviewText'])
		review_star.append(reviews[l]['overall'])
    
	review_set=[]
	review_star_set=[]
	c1=c2=c3=c4=c5=0
	for l in range(0,len(reviews)):
		if(reviews[l]['overall']==5 and c5<5000):
			review_set.append(reviews[l]['reviewText'])
			review_star_set.append(reviews[l]['overall'])
			c5=c5+1
		if(reviews[l]['overall']==4 and c4<5000):
			review_set.append(reviews[l]['reviewText'])
			review_star_set.append(reviews[l]['overall'])
			c4=c4+1
		if(reviews[l]['overall']==3 and c3<3000):
			review_set.append(reviews[l]['reviewText'])
			review_star_set.append(reviews[l]['overall'])
			c3=c3+1
		if(reviews[l]['overall']==2 and c2<5000):
			review_set.append(reviews[l]['reviewText'])
			review_star_set.append(reviews[l]['overall'])
			c2=c2+1
		if(reviews[l]['overall']==1 and c1<5000):
			review_set.append(reviews[l]['reviewText'])
			review_star_set.append(reviews[l]['overall'])
			c1=c1+1
        
	review_sentiment=[]
	#len(review_sentiment)
	p=0
	for l in range(0,len(review_star_set)):
   
		if review_star_set[l]==4.0 or review_star_set[l]==5.0:
        
			p=1
		else: 
			if(review_star_set[l]==3.0):
				p=1
			else:
				if(review_star_set[l]==1.0 or review_star_set[l]==2.0):
					p=0
		review_sentiment.append(p)
		
	return review_set,review_sentiment
    

	

		
if __name__=='__main__':
	hello_python()
		