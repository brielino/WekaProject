package packagem;

import weka.core.Instances;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.lazy.IBk;


public class TestWekaEasy{
	
	public static final String PATH0="\\Users\\gabri\\OneDrive\\Desktop\\Metriche";
	public static final String PATH1="C:";
	public static final String PROJECTNAME="TAJO";
	
	private TestWekaEasy() {
		throw new UnsupportedOperationException();
	}
	
	public static void writeFileCsv(String classifier,int i,double precision,double recall,double auc,double kappa,FileWriter fileWriter) throws IOException {
		fileWriter.append(PROJECTNAME);
		fileWriter.append(",");
		fileWriter.append(Integer.toString(i));
		fileWriter.append(",");
		fileWriter.append(classifier);
		fileWriter.append(",");
		fileWriter.append(Double.toString(precision));
		fileWriter.append(",");
		fileWriter.append(Double.toString(recall));
		fileWriter.append(",");
		fileWriter.append(Double.toString(auc));
		fileWriter.append(",");
		fileWriter.append(Double.toString(kappa));
		fileWriter.append(",");
		fileWriter.append("\n");
		
		
	}
	
	public static void main(String[] args) throws Exception{
		//load datasets
		ArrayList<Integer> version=(ArrayList<Integer>) WekaCreateFileArff.foundVersion(PROJECTNAME);
		try(FileWriter fileWriter = new FileWriter(PATH1+"\\Users\\gabri\\OneDrive\\Desktop\\"+PROJECTNAME+"WekaInfo.csv")){
			fileWriter.append("DataSet,#TrainingR,Classifier,Precision,Recall,AUC,Kappa\n");	
			for(int i = 0;i<version.size();i++) {
				if(version.get(i)!=1) {
					DataSource source1 = new DataSource(PATH1+"\\Users\\gabri\\OneDrive\\Desktop\\WekaFiles\\Metriche"+PROJECTNAME+version.get(i)+"Training.arff");
					Instances training = source1.getDataSet();
					DataSource source2 = new DataSource(PATH1+"\\Users\\gabri\\OneDrive\\Desktop\\WekaFiles\\Metriche"+PROJECTNAME+version.get(i)+"Testing.arff");
					Instances testing = source2.getDataSet();
					int numAttr = training.numAttributes();
					training.setClassIndex(numAttr - 1);
					testing.setClassIndex(numAttr - 1);
					NaiveBayes classifier = new NaiveBayes();
					RandomForest classifier1 =new RandomForest();
					IBk classifier2 = new IBk();
					classifier.buildClassifier(training);
					classifier1.buildClassifier(training);
					classifier2.buildClassifier(training);
					Evaluation eval = new Evaluation(testing);
					eval.evaluateModel(classifier, testing);
					writeFileCsv("NaiveBayes",i,eval.precision(1),eval.recall(1),eval.areaUnderROC(1),eval.kappa(),fileWriter);
					eval.evaluateModel(classifier1, testing);
					writeFileCsv("RandomForest",i,eval.precision(1),eval.recall(1),eval.areaUnderROC(1),eval.kappa(),fileWriter);
					eval.evaluateModel(classifier2, testing);
					writeFileCsv("IBk",i,eval.precision(1),eval.recall(1),eval.areaUnderROC(1),eval.kappa(),fileWriter);
				}
			}
		}
	}
}