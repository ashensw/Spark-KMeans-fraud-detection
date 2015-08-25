import org.apache.spark.api.java.function.Function;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;


/**
 * Created by ashen on 8/21/15.
 */
public class DoubleArrayToVector implements Function<double[], Vector> {

    private static final long serialVersionUID = 8329428281317101710L;

    /**
     * Function to transform double array into labeled point.
     *
     * @param features  Double array of tokens
     * @return          Vector
     */

    @Override
    public Vector call(double[] features) {
        return Vectors.dense(features);

    }
}
