/**
 * Created by ashen on 7/30/15.
 */
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.SparkConf;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;
import org.apache.spark.sql.DataFrame;
import java.util.ArrayList;
import java.util.List;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.commons.math3.ml.distance.EuclideanDistance;
import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;


public class SparkExperimentsRunning {

    //private static final long serialVersionUID = -6377573678240024862L;

    private static int numClusters;
    private static int numIterations = 100;
    private static int[] noOfPointsInClustersArray;
    private static double[] maxDistanceArray;
    private static double[] percentilesArray;
    private static double[] distancesArray;
    private static double[][] distancesOfEachCluster;
    private static Lable[] lablesCount;
    private static List[] lablesByClusters;
    private static Object[][] lablesByClustersArray;
    private static Lable[][] lableCountOfEachClusters;
    private static double[] max;
    private static double[] min;

    private static long[][] evalNormal;
    private static long[][] evalFraud;
    private static long[][] testNormal;
    private static long[][] testFraud;
    private static double[] wsse;



    private static JavaSparkContext createSparkContexts(String appName, String master){
        SparkConf conf = new SparkConf().setAppName(appName).setMaster(master).set("spark.driver.memory", "3g");
        JavaSparkContext sc = new JavaSparkContext(conf);

        return sc;

    }

    public static void main(String[] args){



        //String masterURL = "local";
        //String masterURL = "spark://10.100.1.154:7077";
        String masterURL = "spark://127.0.0.1:7077";
        String appName = "K_Means";


        JavaSparkContext sc = createSparkContexts(appName, masterURL);
        SQLContext sqlContext = new org.apache.spark.sql.SQLContext(sc);

        //loading the dataset into a Dataframe
        DataFrame df = sqlContext.read().format("com.databricks.spark.csv").option("header", "false").load("kddcup.data_10_percent_corrected.csv");

        //filtering and seperate normal and fraud data
        DataFrame normal = df.filter(df.col("C41").like("normal."));

        DataFrame fraud = df.filter(df.col("C41").notEqual("normal."));

        /////// Spliting the datasets //////////

        double[] weights = {0.65,0.15,0.2};

        DataFrame[] normalSplit = normal.randomSplit(weights);
        DataFrame[] fraudSplit = fraud.randomSplit(weights);

        List lableNames = normalSplit[0].select("C41").distinct().collectAsList();
        List lableData = normalSplit[0].select("C41").collectAsList();

        String dropFeatures[] = {"C1","C2","C3","C41"};


        //normal 65% data for train the model
        DataFrame trainDF = dropFeatures(dropFeatures,normalSplit[0]);

        //normal 15% data for optimize the model
        DataFrame evalNormalDF = dropFeatures(dropFeatures,normalSplit[1]);

        //normal 20% data for test the model
        DataFrame testNormalDF = dropFeatures(dropFeatures,normalSplit[2]);

        //fraud 15% data for optimize the model
        DataFrame evalFraudDF = dropFeatures(dropFeatures, fraudSplit[1]);

        //fraud 20% data for test the model
        DataFrame testFraudDF = dropFeatures(dropFeatures, fraudSplit[2]);


        //method to find max and min values of each ferture
        maxAndMin(trainDF);


        //applying Spark Transformations

        JavaRDD<Row> trainRow = trainDF.toJavaRDD();
        JavaRDD<Vector> parsedDataTrain = trainRow.map(new RowsToLines(",")).map(new LinesToStringArray(",")).map(new StringArrayToDoubleArray()).map(new Normalize(max,min)).map(new DoubleArrayToVector());
        parsedDataTrain.cache();

        JavaRDD<Row> evalNormalRow = evalNormalDF.toJavaRDD();
        JavaRDD<Vector> parsedDataEvalNormal = evalNormalRow.map(new RowsToLines(",")).map(new LinesToStringArray(",")).map(new StringArrayToDoubleArray()).map(new Normalize(max,min)).map(new DoubleArrayToVector());
        parsedDataEvalNormal.cache();

        JavaRDD<Row> testNormalRow = testNormalDF.toJavaRDD();
        JavaRDD<Vector> parsedDataTestNormal = testNormalRow.map(new RowsToLines(",")).map(new LinesToStringArray(",")).map(new StringArrayToDoubleArray()).map(new Normalize(max,min)).map(new DoubleArrayToVector());
        parsedDataTestNormal.cache();

        JavaRDD<Row> evalFraudRow = evalFraudDF.toJavaRDD();
        JavaRDD<Vector> parsedDataEvalFraud = evalFraudRow.map(new RowsToLines(",")).map(new LinesToStringArray(",")).map(new StringArrayToDoubleArray()).map(new Normalize(max,min)).map(new DoubleArrayToVector());
        parsedDataEvalFraud.cache();

        JavaRDD<Row> testFraudRow = testFraudDF.toJavaRDD();
        JavaRDD<Vector> parsedDataTestFraud = testFraudRow.map(new RowsToLines(",")).map(new LinesToStringArray(",")).map(new StringArrayToDoubleArray()).map(new Normalize(max,min)).map(new DoubleArrayToVector());
        parsedDataTestFraud.cache();


        int[] numClustersArray = {2,3,4,5,6,7,8,9,10,20,30,40,50,100};

        wsse = new double[numClustersArray.length];

        evalNormal = new long[numClustersArray.length][2];
        evalFraud = new long[numClustersArray.length][2];
        testNormal = new long[numClustersArray.length][2];
        testFraud = new long[numClustersArray.length][2];

        double WSSSE=0;


        for(int i=0; i<numClustersArray.length; i++) {

            numClusters = numClustersArray[i];
            noOfPointsInClustersArray = new int[numClusters];
            maxDistanceArray = new double[numClusters];

            //Training the model
            KMeansModel clusters = KMeans.train(parsedDataTrain.rdd(), numClusters, numIterations);

            //calculate WSSE
            WSSSE = clusters.computeCost(parsedDataTrain.rdd());
            wsse[i] = WSSSE;

            Vector[] centers = clusters.clusterCenters();
            JavaRDD<Integer> cen = clusters.predict(parsedDataTrain);


            List list1 = parsedDataTrain.collect();
            List list2 = cen.collect();

            distancesCalculation(centers, list1, list2);

            separateLalesIntoClusters(list2, lableData);

            separateDistancesIntoClusters(list2);
            calculatePercentiles();

            // printArray(maxDistanceArray);
            System.out.println();
            printArray(noOfPointsInClustersArray);
            System.out.println();
            // printArray(lablesCount);


            countlablesofClusters(lableNames, lablesByClusters);
            print2DArray(lableCountOfEachClusters);

            predictFrauds2(clusters, parsedDataEvalNormal,evalNormal,i);

            predictFrauds2(clusters, parsedDataEvalFraud,evalFraud,i);

            predictFrauds2(clusters, parsedDataTestNormal,testNormal,i);

            predictFrauds2(clusters, parsedDataTestFraud, testFraud, i);


        }

        long total;
        double per;

        System.out.println(" -------- Prediction results ----------");

        for(int i=0; i<numClustersArray.length; i++){

            System.out.println();
            System.out.println("No of Clusters : "+numClustersArray[i]);
            System.out.println("Within Set Sum of Squared Errors = " + wsse[i]);
            System.out.println();
            System.out.println(" -- Evaluation Data --");
            System.out.println();
            System.out.println("Normal Data");
            total = evalNormal[i][0] + evalNormal[i][1];
            per = (evalNormal[i][0]*100)/total;
            System.out.println("Corectly identified :"+evalNormal[i][0]+"  "+per+"%");
            per = (evalNormal[i][1]*100)/total;
            System.out.println("Incorectly identified :"+evalNormal[i][1]+"  "+per+"%");

            System.out.println();
            System.out.println("Fraud Data");
            total = evalFraud[i][0] + evalFraud[i][1];
            per = (evalFraud[i][1]*100)/total;
            System.out.println("Corectly identified :"+evalFraud[i][1]+"  "+per+"%");
            per = (evalFraud[i][0]*100)/total;
            System.out.println("Incorectly identified :"+evalFraud[i][0]+"  "+per+"%");
            System.out.println();

            System.out.println(" -- Test Data --");

            System.out.println();
            System.out.println("Normal Data");
            total = testNormal[i][0] + testNormal[i][1];
            per = (testNormal[i][0]*100)/total;
            System.out.println("Corectly identified :"+testNormal[i][0]+"  "+per+"%");
            per = (testNormal[i][1]*100)/total;
            System.out.println("Incorectly identified :"+testNormal[i][1]+"  "+per+"%");

            System.out.println();
            System.out.println("Fraud Data");
            total = testFraud[i][0] + testFraud[i][1];
            per = (testFraud[i][1]*100)/total;
            System.out.println("Corectly identified :"+testFraud[i][1]+"  "+per+"%");
            per = (testFraud[i][0]*100)/total;
            System.out.println("Incorectly identified :"+testFraud[i][0]+"  "+per+"%");


        }


    }

    public static DataFrame dropFeatures(String features[], DataFrame df){
        for(int i=0; i< features.length; i++) {
            df = df.drop(features[i]);
        }
        return df;

    }

    public static void maxDistance(int cluster, double newValue){

        //double newMax = 0;

        if(maxDistanceArray[cluster]<newValue)
            maxDistanceArray[cluster] = newValue;


    }

    public static void noOfPointsInClusters(int cluster){

        noOfPointsInClustersArray[cluster]++;

    }

    public static void distancesCalculation(Vector[] clusterCenters, List dataList, List clusterList){

        distancesArray = new double[dataList.size()];
        EuclideanDistance distance = new EuclideanDistance();

        Vector dataPoint = null;
        int center = 0;

        for(int i=0; i<dataList.size() ; i++){

            dataPoint = (Vector)dataList.get(i);
            center = (Integer)clusterList.get(i);

            distancesArray[i] = distance.compute(clusterCenters[center].toArray(),dataPoint.toArray());
//            maxDistance(center,distancesArray[i]);
            noOfPointsInClusters(center);
        }
        System.out.println();
        System.out.println("Finished distacesCalculation()");
        System.out.println();
    }


    public static void countlables(List lables, List data){

        lablesCount = new Lable[lables.size()];

        for(int i=0; i<lables.size(); i++){
            lablesCount[i] = new Lable(lables.get(i).toString());
        }

        for(int j=0; j<data.size(); j++){

            for(int k=0; k<lablesCount.length; k++){
                if(lablesCount[k].getName().equalsIgnoreCase(data.get(j).toString())){
                    lablesCount[k].updateCount();
                    break;
                }
            }

        }
    }

    public static void countlablesofClusters(List lables, List[] data){

        lableCountOfEachClusters = new Lable[numClusters][lables.size()];

        for(int i=0; i<numClusters; i++){
            for(int j=0; j<lables.size(); j++){
                lableCountOfEachClusters[i][j] = new Lable(lables.get(j).toString());
            }

        }

        for(int m=0; m<numClusters; m++){

            for(int k=0; k<data[m].size(); k++){

                for(int l=0; l<lables.size(); l++){
                    if(lableCountOfEachClusters[m][l].getName().equalsIgnoreCase(data[m].get(k).toString())){
                        lableCountOfEachClusters[m][l].updateCount();
                        break;
                    }
                }

            }

        }

    }

    public static void separateLalesIntoClusters(List centers, List data){


        lablesByClusters = new List[numClusters];
        for(int m=0; m<numClusters; m++){
            lablesByClusters[m] = new ArrayList<String>();

        }
        int tempCluster;
        int tempIndex;
        for(int i=0; i<centers.size(); i++){

            tempCluster = (Integer)centers.get(i);
            tempIndex = lablesByClusters[tempCluster].size();

            lablesByClusters[tempCluster].add(tempIndex,data.get(i));
        }

    }

    public static void separateDistancesIntoClusters(List centers){

        distancesOfEachCluster = new double[numClusters][];

        int center;
        int[] counts = new int[numClusters];

        for(int i=0; i<numClusters; i++){
            distancesOfEachCluster[i] = new double[noOfPointsInClustersArray[i]];
            counts[i] = 0;
        }

        for(int j=0; j<centers.size(); j++){
            center = (Integer)centers.get(j);
            distancesOfEachCluster[center][counts[center]] = distancesArray[j];
            counts[center]++;
        }
    }

    public static void calculatePercentiles(){

        percentilesArray = new double[numClusters];

        // Get a DescriptiveStatistics instance
        DescriptiveStatistics stats = new DescriptiveStatistics();


        for(int i=0; i<numClusters; i++){

            // Add the data from the array
            for( int j = 0; j < distancesOfEachCluster[i].length; j++) {
                stats.addValue(distancesOfEachCluster[i][j]);
            }

            // Compute some statistics
//        double mean = stats.getMean();
//        double std = stats.getStandardDeviation();

            percentilesArray[i] = stats.getPercentile(97);
            System.out.println("before : "+percentilesArray[i]);
            //System.out.println(stats.getSkewness());
            stats.clear();
            System.out.println("after : " + percentilesArray[i]);
        }


    }

    public static boolean predictNewDataPoint(Vector datapoint, KMeansModel cluster){

        int predictedCluster = cluster.predict(datapoint);
        EuclideanDistance distanceE = new EuclideanDistance();
        Vector[] clusterCenters = cluster.clusterCenters();
        double distance =  distanceE.compute(clusterCenters[predictedCluster].toArray(), datapoint.toArray());

        boolean fraud;
        if (distance > percentilesArray[predictedCluster]){
            fraud = true;
        }
        else{
            fraud = false;
        }

        return fraud;
    }


    public static void predictFrauds(KMeansModel cluster, JavaRDD<Vector> testData){

        Vector[] clusterCenters = cluster.clusterCenters();

        JavaRDD<Integer> centers = cluster.predict(testData);
        List centerList = centers.collect();
        List data = testData.collect();

        EuclideanDistance distanceE = new EuclideanDistance();

        //long totalData = testData.count();
        long normal = 0;
        long frauds = 0;
        Vector dataPoint = null;
        int centerIndex = 0;
        double distance = 0;

        for(int i=0; i<centerList.size(); i++){

             dataPoint = (Vector)data.get(i);
             centerIndex = (Integer)centerList.get(i);

             distance = distanceE.compute(clusterCenters[centerIndex].toArray(), dataPoint.toArray());

            if(distance > percentilesArray[centerIndex]){
                frauds++;
            }
            else{
                normal++;
            }
        }

        long total = normal+frauds;
        double nP = normal*100/total;
        double fP = frauds*100/total;

        System.out.println("Normal count : "+normal+"  :"+nP+"%");
        System.out.println("Frauds count : "+frauds+"  :"+fP+"%");
    }

    public static void predictFrauds2(KMeansModel cluster, JavaRDD<Vector> testData, long[][] dataArray, int index){

        Vector[] clusterCenters = cluster.clusterCenters();

        JavaRDD<Integer> centers = cluster.predict(testData);
        List centerList = centers.collect();
        List data = testData.collect();

        EuclideanDistance distanceE = new EuclideanDistance();

        //long totalData = testData.count();
        long normal = 0;
        long frauds = 0;
        Vector dataPoint = null;
        int centerIndex = 0;
        double distance = 0;

        for(int i=0; i<centerList.size(); i++){

            dataPoint = (Vector)data.get(i);
            centerIndex = (Integer)centerList.get(i);

            distance = distanceE.compute(clusterCenters[centerIndex].toArray(), dataPoint.toArray());

            if(distance > percentilesArray[centerIndex]){
                frauds++;
            }
            else{
                normal++;
            }
        }

        dataArray[index][0] = normal;
        dataArray[index][1] = frauds;

//        long total = normal+frauds;
//        double nP = normal*100/total;
//        double fP = frauds*100/total;
//
//        System.out.println("Normal count : "+normal+"  :"+nP+"%");
//        System.out.println("Frauds count : "+frauds+"  :"+fP+"%");


    }


    public static void seperateLalesIntoClusters2(List centers, List data){


        lablesByClustersArray = new Object[numClusters][];
        int[] tempCount = new int[numClusters];
        for(int i=0; i<numClusters; i++){
            lablesByClustersArray = new Object[i][noOfPointsInClustersArray[i]];
        }
        for(int i=0; i<data.size(); i++){

            lablesByClustersArray[(Integer)centers.get(i)][tempCount[(Integer)centers.get(i)]] = data.get(i);
            tempCount[(Integer)centers.get(i)]++;
        }

    }

    public static void printArray(double[] array){

        for(int i=0; i<array.length; i++){
            System.out.println("Value "+i+" : "+array[i]);
        }
    }

    public static void printArray(int[] array){

        for(int i=0; i<array.length; i++){
            System.out.println("Value "+i+" : "+array[i]);
        }
    }

    public static void printArray(Lable[] array){

        for(int i=0; i<array.length; i++){
            System.out.println("Value "+i+" : "+array[i]);
        }
    }

    public static void print2DArray(Lable[][] array){

        for(int i=0; i<array.length; i++){
            System.out.println("\nCluster : "+i+" summery\n");
            for(int j=0; j<array[i].length; j++){
                System.out.println("Value "+i+" : "+array[i][j]);
            }
        }
    }

    public static void maxAndMin(DataFrame df){

        String[] features = df.columns();
        max = new double[features.length];
        min = new double[features.length];
        long count = df.count();
        double value =0;

        // Get a DescriptiveStatistics instance
        DescriptiveStatistics stats2 = new DescriptiveStatistics();

        //DataFrame d;
        Row[] row = df.collect();


        for(int i=0; i<features.length; i++ ){

            for(int j=0; j<count; j++) {

                value = Double.parseDouble((String)row[j].get(i));

                // Add the data from the array
                stats2.addValue(value);

            }

            max[i] = stats2.getMax();
            min[i] = stats2.getMin();

            System.out.println("max : "+max[i]);
            System.out.println("min : "+min[i]);

            stats2.clear();

        }
    }


}

