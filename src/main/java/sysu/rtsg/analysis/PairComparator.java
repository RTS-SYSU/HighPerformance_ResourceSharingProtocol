package sysu.rtsg.analysis;

import java.util.Comparator;

public class PairComparator<T1 extends Comparable<T1>, T2> implements Comparator<Pair<T1, T2>> {
    @Override
    public int compare(Pair<T1, T2> p1, Pair<T1, T2> p2) {
        // 从大到小排序
        return p2.getKey().compareTo(p1.getKey());
    }
}
