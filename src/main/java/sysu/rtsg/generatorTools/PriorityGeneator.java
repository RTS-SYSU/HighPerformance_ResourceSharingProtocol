package sysu.rtsg.generatorTools;

import sysu.rtsg.entity.SporadicTask;

import java.util.ArrayList;

public class PriorityGeneator {
	public static final int MAX_PRIORITY = 1000;

	public ArrayList<ArrayList<SporadicTask>> assignPrioritiesByDM(ArrayList<ArrayList<SporadicTask>> tasksToAssgin) {
		if (tasksToAssgin == null) {
			return null;
		}

		ArrayList<ArrayList<SporadicTask>> tasks = new ArrayList<>(tasksToAssgin);
		for (int i = 0; i < tasks.size(); i++) {
			new PriorityGeneator().deadlineMonotonicPriorityAssignment(tasks.get(i), tasks.get(i).size());
		}

		return tasks;
	}

	private void deadlineMonotonicPriorityAssignment(ArrayList<SporadicTask> taskset, int NoT) {
		ArrayList<Integer> priorities = generatePriorities(NoT);
		/* deadline monotonic assignment */
		taskset.sort((t1, t2) -> Double.compare(t1.deadline, t2.deadline));
		priorities.sort((p1, p2) -> -Integer.compare(p1, p2));
		for (int i = 0; i < taskset.size(); i++) {
			taskset.get(i).priority = priorities.get(i);
		}
	}

	private ArrayList<Integer> generatePriorities(int number) {
		ArrayList<Integer> priorities = new ArrayList<>();
		for (int i = 0; i < number; i++)
			priorities.add(MAX_PRIORITY - (i + 1) * 2);
		return priorities;
	}
}
