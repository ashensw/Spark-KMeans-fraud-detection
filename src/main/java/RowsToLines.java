/**
 * Created by ashen on 8/4/15.
 */
import org.apache.spark.api.java.function.Function;
import org.apache.spark.sql.Row;

/**
 * This class converts Spark SQL Rows into a CSV/TSV string.
 */
public class RowsToLines implements Function<Row, String> {

    private static final long serialVersionUID = -5025419727399292773L;
    private String columnSeparator;

    public RowsToLines(String columnSeparator) {
        this.columnSeparator = columnSeparator;
    }

    @Override
    public String call(Row row) {
        StringBuilder sb = new StringBuilder();
        if (row.length() <= 0) {
            return "";
        }
        for (int i = 0; i < row.length(); i++) {
            sb.append(row.get(i) + columnSeparator);
        }
        return sb.substring(0, sb.length() - 1);
    }

}
