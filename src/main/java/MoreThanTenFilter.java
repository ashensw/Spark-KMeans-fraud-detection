import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.mllib.regression.LabeledPoint;

public class MoreThanTenFilter implements VoidFunction<LabeledPoint> {

    @Override
    public void call(LabeledPoint point) throws Exception {
        try {
            if (point.label() > 10) {
                System.out.println();
                System.out.println("You have a label greater than 10");
                System.out.println("You have a label greater than 10");
                System.out.println("You have a label greater than 10");
                System.out.println();
            }

        } catch (Exception e) {
            throw new Exception("error");
        }
    }
}