import org.apache.spark.api.java.function.Function;
import org.apache.spark.sql.Row;


/**
 * Created by ashen on 8/21/15.
 */
public class LinesToStringArray implements Function<String, String[]> {

    private static final long serialVersionUID = 8329428281317101710L;
    private String columnSeparator;

    public LinesToStringArray(String columnSeparator) {
        this.columnSeparator = columnSeparator;
    }

    @Override
    public String[] call(String line) {
        String[] sarray = line.split(",");

        return sarray;
    }
}
