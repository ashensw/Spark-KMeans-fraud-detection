import org.apache.spark.api.java.function.Function;

/**
 * Created by ashen on 8/21/15.
 */
public class StringArrayToDoubleArray implements Function<String[], double[]> {

    private static final long serialVersionUID = 8329428281317101710L;

    @Override
    public double[] call(String[] sarray) {
        double[] values = new double[sarray.length];

        for (int i = 0; i < sarray.length; i++) {

            if(!isNumeric(sarray[i])) {
                continue;
            }
            values[i] = Double.parseDouble(sarray[i]);
        }

        return values;

    }

    public static boolean isNumeric(String str)
    {
        try
        {
            double d = Double.parseDouble(str);
        }
        catch(NumberFormatException nfe)
        {
            return false;
        }
        return true;
    }
}
