from ant_colony_my import ant_colony
import sys
import time

def create_classifiers():
	from sklearn import svm
	from sklearn.neighbors import KNeighborsClassifier
	from sklearn.neural_network import MLPClassifier									
	from sklearn.ensemble import AdaBoostClassifier
	from sklearn.linear_model import SGDClassifier
	from sklearn.naive_bayes import MultinomialNB
	from sklearn.tree import DecisionTreeClassifier
	from sklearn.linear_model import LogisticRegression
	from sklearn.ensemble import RandomForestClassifier
	
	print("Preparing Classifiers...")
	rfc = RandomForestClassifier(random_state=1)
	lr = LogisticRegression(random_state=1)
	dt=DecisionTreeClassifier( criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_split=0, class_weight=None, presort=False)
	dt2=DecisionTreeClassifier( criterion='entropy', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_split=0, class_weight=None, presort=False)

	nb=MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
	sgd=SGDClassifier()
	ada= AdaBoostClassifier()
	mlp=MLPClassifier()
	knn=KNeighborsClassifier(n_neighbors=5, weights='distance', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=1)
	sv_rbf=svm.SVC(C=10.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None)
	sv_poly=svm.SVC(C=10.0, kernel='poly', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None)
	sv_linear=svm.SVC(C=10.0, kernel='linear', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None)

	#list_of_classifier={'svm_rbf':sv_rbf, 'knn':knn, 'mlp':mlp, 'ada':ada, 'sgd': sgd,'mnb': nb, 'dt':dt,'dt2':dt2,'lr':lr, 'rfc':rfc, 'svm_poly':sv_poly,'svm_linear':sv_linear}
	list_of_classifier={'svm_rbf':sv_rbf, 'knn':knn, 'mlp':mlp, 'ada':ada, 'sgd': sgd,'mnb': nb, 'dt':dt,'lr':lr, 'rfc':rfc}

	#pheromone_map=create_pheromone_list(list_of_classifier)
	return list_of_classifier
	
	review_set,review_sentiment=create_dataset()
	#ant_c=ant_colony(list_of_classifier,review_set,review_sentiment)
	#ant_c.mainloop()
	
def create_dataset(FileName):

	import json
	print("Preparing dataset...")
	reviews = []
	for line in open(FileName, 'r'):
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
	
def print_results(current_classifiers,X_train_dtm,y_train,X_test_dtm,y_test):
	
	
	from sklearn.metrics import classification_report
	from sklearn import metrics
	from sklearn import model_selection
	seed = 7
	estimator=[]
	for key, value in current_classifiers.items():
		estimator.append((key,value))
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	from sklearn.ensemble import VotingClassifier
	vot_1 = VotingClassifier(estimators=estimator,voting='hard')
	vot_1=vot_1.fit(X_train_dtm,y_train)
	vot_1_pred=vot_1.predict(X_test_dtm)
	acc_vot_1=metrics.accuracy_score(y_test,vot_1_pred)
	print('voting classifier:')
	print('Vot_1 Accuracy: ',acc_vot_1)
	report_vot_1 = classification_report(y_test,vot_1_pred)
	print(report_vot_1)
	"""results = model_selection.cross_val_score(vot_1, X_train_dtm, y_train, cv=kfold)
	print("Cross-validation result: ",results.mean())"""
	
	
	
if __name__=='__main__':
	f = open("test_init_3classifier_9_10_10_baby.out", 'w')
	sys.stdout = f
	start_time=time.clock()
	print('start_time: ',start_time)
	import progressbar
	from time import sleep
	bar = progressbar.ProgressBar(maxval=20, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
	bar.start()
	current_classifiers2={}
	list_of_classifier=create_classifiers()
	#review_set,review_sentiment=create_dataset('DataSet/Cell_Phones_and_Accessories_5.json')
	#review_set,review_sentiment=create_dataset('DataSet/Digital_Music_5.json')
	#review_set,review_sentiment=create_dataset('DataSet/Health_and_Personal_Care_5.json')
	#review_set,review_sentiment=create_dataset('DataSet/Beauty_5.json')
	#review_set,review_sentiment=create_dataset('DataSet/Pet_Supplies_5.json')
	#review_set,review_sentiment=create_dataset('DataSet/Sports_and_Outdoors_5.json')
	review_set,review_sentiment=create_dataset('DataSet/Baby_5.json')
	#review_set,review_sentiment=create_dataset('DataSet/Office_Products_5.json')
	#review_set,review_sentiment=create_dataset('DataSet/Clothing_Shoes_and_Jewelry_5.json')
	#review_set,review_sentiment=create_dataset('DataSet/Grocery_and_Gourmet_Food_5.json')
	#review_set,review_sentiment=create_dataset('DataSet/Tools_and_Home_Improvement_5.json')
	
	
	
	
	
	import sklearn
	from sklearn.cross_validation import train_test_split
	X_train, X_test, y_train, y_test=train_test_split(review_set,review_sentiment,random_state=1)
	from sklearn.feature_extraction.text import CountVectorizer
	vect=CountVectorizer(stop_words='english',ngram_range=(1, 2),max_features=500)
	vect.fit(X_train)
	X_train_dtm=vect.transform(X_train)
	X_test_dtm=vect.transform(X_test)
	
	
	ant_c=ant_colony(list_of_classifier,review_set,review_sentiment)
	ant_c.mainloop()
	for k in range(ant_c.ant_count):
		print('Result for ensemble classifier: ',ant_c.ants[k].current_classifiers)
		for key in ant_c.ants[k].current_classifiers:
			current_classifiers2[key]=list_of_classifier[key]
		print_results(current_classifiers2,X_train_dtm,y_train,X_test_dtm,y_test)
	end_time=time.clock()
	total_time=(end_time-start_time)/60
	print('Time(in min) taken to execute: ',total_time)
	print()
	f.close()
	bar.finish()
	