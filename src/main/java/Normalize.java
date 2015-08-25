import org.apache.spark.api.java.function.Function;

/**
 * Created by ashen on 8/21/15.
 */
public class Normalize implements Function<double[], double[]> {

    private static final long serialVersionUID = 8329428281317101710L;
    private double[] max;
    private double[] min;
    int temp;



    public Normalize(double[] max, double[] min){

        this.max = max;
        this.min = min;
    }

    @Override
    public double[] call(double[] values) {
        double[] normalizedValues = new double[values.length];

        for (int i = 0; i < values.length; i++) {

            temp = i % max.length;

            if(min[temp] != max[temp]){
                normalizedValues[i] = (values[i] - min[temp])/(max[temp] - min[temp]);
            }
            else if(min[temp] == 0 && max[temp] == 0 ){
                normalizedValues[i] = 0;
            }
            else{
                normalizedValues[i] = 0.5;
            }

        }

        return normalizedValues;
    }
}
