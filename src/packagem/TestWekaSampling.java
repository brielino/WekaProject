package packagem;

import weka.core.Instances;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.RandomForest;
import weka.filters.supervised.instance.Resample;
import weka.filters.supervised.instance.SpreadSubsample;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.lazy.IBk;
import weka.filters.supervised.instance.SMOTE;
/*Classe che mi permettee e di applicare Sampling su un Data Set 66/33 
 * applicando 3 tipi di Classificatori e diverse metodologie di Sampling 
 */
public class TestWekaSampling {
	
	public static final String PATH="\\Users\\gabri\\OneDrive\\Desktop\\Metriche";
	public static final String PATH0="\\Users\\gabri\\OneDrive\\Desktop\\SamplingFiles\\";
	public static final String PATH1="C:";
	public static final String PROJECTNAME="TAJO";
	public static final String TRAININGSMOTE="TrainingSmote.csv";
	public static final String RANDOMFOREST="RandomForest";
	public static final String NAIVEBAYES="NaiveBayes";
	public static final String IBK="IBk";
	public static final String SMOTE="SMOTE";
	public static final String NS="NoSampling";
	public static final String OVS="OverSampling";
	public static final String UNS="UnderSampling";
	protected static final String[]  OPTS={ "-B", "1.0", "-Z", "130.3"};
	
	private TestWekaSampling() {
		throw new UnsupportedOperationException();
	}

	public static int foundPartition() throws IOException {
		/* Calcolo il numero di istanze per una partizione 66/33% (Training/Testing) */
		String line="";
		int count=0;
		try(BufferedReader filecsv= new BufferedReader(new FileReader(PATH1+PATH+PROJECTNAME+".csv"))){
			line=filecsv.readLine();
			count++;
			while(line!=null) {
				count++;
				line=filecsv.readLine();
				
			}
		}
		return count*66/100;
	}
	
	
	public static void createFileCsvTraining(int numberSplit) throws IOException {
		/* Creo 2 file Csv uno per il Testing e un per Training utilizzando numSplit come delimitatore per 
		 * la partizione del Data Set
		 */
		String line="";
		int count=1;
		try(BufferedReader filecsv= new BufferedReader(new FileReader(PATH1+PATH+PROJECTNAME+".csv"))){
			try(FileWriter fileTraining=new FileWriter(PATH1+PATH0+PROJECTNAME+TRAININGSMOTE);
					FileWriter fileTesting =new FileWriter(PATH1+PATH0+PROJECTNAME+"TestingSmote.csv")){
				line=filecsv.readLine();
				fileTraining.write(line);
				fileTraining.write("\n");
				fileTesting.write(line);
				fileTesting.write("\n");
				line=filecsv.readLine();
				while(count!=numberSplit) {
					line=filecsv.readLine();
					fileTraining.write(line);
					fileTraining.write("\n");
					count++;
				}
				line=filecsv.readLine();
				while(line!=null) {
					fileTesting.write(line);
					fileTesting.write("\n");
					line=filecsv.readLine();
				}
			}
		}	
	}
	public static double takeYforOverSampling() throws IOException {
		/* mi permette di calcolare la percentuale della classe maggioritara del Data Set,
		 * informaazione necessarie per poter implementare OverSampling 
		 */
		String line="";
		int count=1;
		int y=0;
		int n=0;
		try(BufferedReader fileTraining =new BufferedReader(new FileReader(PATH1+PATH0+PROJECTNAME+TRAININGSMOTE))){
			line=fileTraining.readLine();
			while(line!=null) {
				String[] z= line.split(",");
				if(z[11].compareTo("YES")==0) {
					y++;
				}
				if(z[11].compareTo("NO")==0) {
					n++;
				}
				count++;
				line=fileTraining.readLine();
			}
		}
		if(y>n) {
			return ((double)y/count);
		}else {
			return ((double)n/count);
		}
	}
	/* Insieme di metodi che implementano le diverse tipologie di Sampling (SMOTE,OverSampling,UnderSampling) 
	 * applicati a i diversi classificatori 
	 */
	public static Evaluation overSamplingRandomForest(Instances training,Instances testing) throws Exception {
		Resample resample = new Resample();
		resample.setInputFormat(training);
		resample.setNoReplacement(false);
		resample.setSampleSizePercent(2*takeYforOverSampling());
		resample.setOptions(OPTS);
		
		FilteredClassifier fc = new FilteredClassifier();

		RandomForest rf = new RandomForest();
		fc.setClassifier(rf);
		
		fc.setFilter(resample);
		fc.buildClassifier(training);
		Evaluation eval2 = new Evaluation(testing);	
		eval2.evaluateModel(fc, testing);
		return eval2;
	}
	
	public static Evaluation overSamplingNaiveBayes(Instances training,Instances testing) throws Exception {
		Resample resample = new Resample();
		resample.setInputFormat(training);
		resample.setNoReplacement(false);
		resample.setSampleSizePercent(2*takeYforOverSampling());
		resample.setOptions(OPTS);
		
		FilteredClassifier fc = new FilteredClassifier();

		NaiveBayes naiveB = new NaiveBayes();
		fc.setClassifier(naiveB);
		
		fc.setFilter(resample);
		fc.buildClassifier(training);
		Evaluation eval2 = new Evaluation(testing);	
		eval2.evaluateModel(fc, testing);
		return eval2;
	}
	
	public static Evaluation overSamplingIBk(Instances training,Instances testing) throws Exception {
		Resample resample = new Resample();
		resample.setInputFormat(training);
		resample.setNoReplacement(false);
		resample.setSampleSizePercent(2*takeYforOverSampling());
		resample.setOptions(OPTS);
		
		FilteredClassifier fc = new FilteredClassifier();

		IBk ibk = new IBk();
		fc.setClassifier(ibk);
		
		fc.setFilter(resample);
		fc.buildClassifier(training);
		Evaluation eval2 = new Evaluation(testing);	
		eval2.evaluateModel(fc, testing);
		return eval2;
	}
	
	public static Evaluation smoteSamplingRandomForest(Instances training,Instances testing) throws Exception {
		FilteredClassifier fc = new FilteredClassifier();
		RandomForest rf = new RandomForest();
		fc.setClassifier(rf);
		
		SMOTE smote = new SMOTE();
		smote.setInputFormat(training);
		fc.setFilter(smote);
		
		fc.buildClassifier(training);
		Evaluation eval2 = new Evaluation(testing);	
		eval2.evaluateModel(fc, testing); //sampled
		return eval2;
	}
	
	public static Evaluation smoteSamplingNaiveBayes(Instances training,Instances testing) throws Exception {
		FilteredClassifier fc = new FilteredClassifier();
		NaiveBayes naiveB = new NaiveBayes();
		fc.setClassifier(naiveB);
		
		SMOTE smote = new SMOTE();
		smote.setInputFormat(training);
		fc.setFilter(smote);
		
		fc.buildClassifier(training);
		Evaluation eval2 = new Evaluation(testing);	
		eval2.evaluateModel(fc, testing); //sampled
		return eval2;
	}
	
	public static Evaluation smoteSamplingIBk(Instances training,Instances testing) throws Exception {
		FilteredClassifier fc = new FilteredClassifier();
		IBk ibk = new IBk();
		fc.setClassifier(ibk);
		
		SMOTE smote = new SMOTE();
		smote.setInputFormat(training);
		fc.setFilter(smote);
		
		fc.buildClassifier(training);
		Evaluation eval2 = new Evaluation(testing);	
		eval2.evaluateModel(fc, testing); //sampled
		return eval2;
	}
	
	public static Evaluation underSamplingRandomForest(Instances training,Instances testing) throws Exception {
		FilteredClassifier fc = new FilteredClassifier();
		RandomForest rf = new RandomForest();
		fc.setClassifier(rf);
		
		SpreadSubsample  spreadSubsample = new SpreadSubsample();
		String[] opts = new String[]{ "-M", "1.0"};
		spreadSubsample.setOptions(opts);
		fc.setFilter(spreadSubsample);
		
		fc.buildClassifier(training);
		Evaluation eval2 = new Evaluation(testing);	
		eval2.evaluateModel(fc, testing); //sampled
		return eval2;
	}
	
	public static Evaluation underSamplingNaiveBayes(Instances training,Instances testing) throws Exception {
		FilteredClassifier fc = new FilteredClassifier();
		NaiveBayes nb = new NaiveBayes();
		fc.setClassifier(nb);
		
		SpreadSubsample  spreadSubsample = new SpreadSubsample();
		String[] opts = new String[]{ "-M", "1.0"};
		spreadSubsample.setOptions(opts);
		fc.setFilter(spreadSubsample);
		
		fc.buildClassifier(training);
		Evaluation eval2 = new Evaluation(testing);	
		eval2.evaluateModel(fc, testing); //sampled
		return eval2;
	}
	
	public static Evaluation underSamplingIBk(Instances training,Instances testing) throws Exception {
		FilteredClassifier fc = new FilteredClassifier();
		IBk ibk = new IBk();
		fc.setClassifier(ibk);
		
		SpreadSubsample  spreadSubsample = new SpreadSubsample();
		String[] opts = new String[]{ "-M", "1.0"};
		spreadSubsample.setOptions(opts);
		fc.setFilter(spreadSubsample);
		
		fc.buildClassifier(training);
		Evaluation eval2 = new Evaluation(testing);	
		eval2.evaluateModel(fc, testing); //sampled
		return eval2;
	}
	
	public static void createFileArff() throws IOException {
		/* Metodo per la conversione di file Csv in file ARFF */
		CSVLoader loader1 = new CSVLoader();
		loader1.setSource(new File(PATH1+PATH0+PROJECTNAME+TRAININGSMOTE));
		Instances data1 = loader1.getDataSet(); 
		ArffSaver saver1 = new ArffSaver();
	    saver1.setInstances(data1);
	    saver1.setFile(new File(PATH1+PATH0+PROJECTNAME+"TrainingSmote.arff"));
	    saver1.writeBatch();
	    
	    CSVLoader loader2 = new CSVLoader();
		loader2.setSource(new File(PATH1+PATH0+PROJECTNAME+"TestingSmote.csv"));
		Instances data2 = loader1.getDataSet();
		ArffSaver saver2 = new ArffSaver();
	    saver2.setInstances(data2);
	    saver2.setFile(new File(PATH1+PATH0+PROJECTNAME+"TestingSmote.arff"));
	    saver2.writeBatch();
	}
	
	public static void writefile(FileWriter filewriter,Evaluation eval,String sampling,String classifier) throws IOException {
		/* Scrive il FileWriter con le informazioni passate come parametri */
		filewriter.append(PROJECTNAME);
		filewriter.append(",");
		filewriter.append(sampling);
		filewriter.append(",");
		filewriter.append(classifier);
		filewriter.append(",");
		filewriter.append(Double.toString(eval.precision(1)));
		filewriter.append(",");
		filewriter.append(Double.toString(eval.recall(1)));
		filewriter.append(",");
		filewriter.append(Double.toString(eval.areaUnderROC(1)));
		filewriter.append(",");
		filewriter.append(Double.toString(eval.kappa()));
		filewriter.append(",");
		filewriter.append(Double.toString(eval.pctCorrect()));
		filewriter.append("\n");		
	}
	
	public static void main(String[] args) throws Exception{
		DataSource source1 = new DataSource(PATH1+PATH0+PROJECTNAME+"TrainingSmote.arff");
		Instances training = source1.getDataSet();
		DataSource source2 = new DataSource(PATH1+PATH0+PROJECTNAME+"TestingSmote.arff");
		Instances testing = source2.getDataSet();
		int numAttr = training.numAttributes();
		training.setClassIndex(numAttr - 1);
		testing.setClassIndex(numAttr - 1);
		try(FileWriter filewriter=new FileWriter(PATH1+PATH0+PROJECTNAME+"InfoSampling.csv")){
			filewriter.append("DataSet,#TipeSampling,Classifier,Precision,Recall,AUC,Kappa,Correct%\n");
			RandomForest rf = new RandomForest();
			rf.buildClassifier(training);
			Evaluation eval = new Evaluation(testing);	
			eval.evaluateModel(rf, testing);
			writefile(filewriter,eval,NS,RANDOMFOREST);
			writefile(filewriter,underSamplingRandomForest(training,testing),UNS,RANDOMFOREST);
			writefile(filewriter,overSamplingRandomForest(training,testing),OVS,RANDOMFOREST);
			writefile(filewriter,smoteSamplingRandomForest(training,testing),SMOTE,RANDOMFOREST);
			NaiveBayes naiveB = new NaiveBayes();
			naiveB.buildClassifier(training);
			Evaluation eval1 = new Evaluation(testing);	
			eval1.evaluateModel(naiveB, testing);
			writefile(filewriter,eval1,NS,NAIVEBAYES);
			writefile(filewriter,underSamplingNaiveBayes(training,testing),UNS,NAIVEBAYES);
			writefile(filewriter,overSamplingNaiveBayes(training,testing),OVS,NAIVEBAYES);
			writefile(filewriter,smoteSamplingNaiveBayes(training,testing),SMOTE,NAIVEBAYES);
			IBk ibk = new IBk();
			ibk.buildClassifier(training);
			Evaluation eval2 = new Evaluation(testing);
			writefile(filewriter,eval2,NS,IBK);
			writefile(filewriter,underSamplingIBk(training,testing),UNS,IBK);
			writefile(filewriter,overSamplingIBk(training,testing),OVS,IBK);
			writefile(filewriter,smoteSamplingIBk(training,testing),SMOTE,IBK);	
		}			
	}
}