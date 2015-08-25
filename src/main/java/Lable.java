/**
 * Created by ashen on 8/12/15.
 */
public class Lable {

    private int count;
    private String name;

    public Lable(String name) {
        this.name = name;
        this.count = 0;
    }

    public String getName() {
        return name;
    }

    public int getCount() {
        return count;
    }

    public void setName(String name) {
        this.name = name;
    }

    public void setCount(int count) {
        this.count = count;
    }

    public void updateCount(){
        this.count++;
    }

    public String toString(){
        return this.getName() + " : " + this.getCount();
    }
}
