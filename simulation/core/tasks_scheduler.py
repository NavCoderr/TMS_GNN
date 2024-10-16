import time
import threading

class TasksScheduler:
    def __init__(self, executorsManager, queueOptimizer):
        self.__executorsManager = executorsManager
        self.__queueOptimizer = queueOptimizer
        self.__jobsDict = None
        self.__tasksSources = dict()
        self.__queueProcessingThread = None
        self.__executorsManagerThread = None
        self.__killed = False
        self.__idle = False
        self.__started = False
        self.__tasksGuard = False

    def __processQueue(self):
        self.__started = True
        while not self.__killed:
            # Use the GNN to optimize the queue
            self.optimizeQueue(iterations=3)  # Adjust iterations for more optimization runs if needed
            self.dispatchTasks()

            if self.__executorsManagerThread is not None:
                if not self.__executorsManagerThread.is_alive():
                    print("Process executor thread dead!", flush=True)

            time.sleep(1)

    def __processExecutors(self):
        try:
            while not self.__killed:
                self.__executorsManager.refreshExecutors()
                self.__executorsManager.performRequests()
                time.sleep(1)
        except Exception as e:
            print("Process executors exception! : {}".format(str(e)), flush=True)

    # Public method for queue optimization
    def optimizeQueue(self, iterations):
        # This calls the GNN-based optimization in QueueOptimizer
        return self.__queueOptimizer.optimizeQueue(iterations)

    # Dispatch tasks based on the GNN-optimized sequence and path evaluation
    def dispatchTasks(self):
        free_executors = self.__executorsManager.freeExecutorsNumber()
        for _ in range(0, free_executors):
            if not self.__queueOptimizer.queue().empty():
                task = self.__queueOptimizer.queue().nextTask()

                # Get the best path for the task using the GNN-enhanced multi-path evaluation
                source, destination = task.getSource(), task.getDestination()
                k_shortest_paths = self.__queueOptimizer._QueueOptimizer__system.graph.get_k_shortest_paths(source, destination, k=3)

                # Evaluate paths using GNN
                best_path = None
                min_cost = float('inf')
                for path in k_shortest_paths:
                    path_tensor = self.__queueOptimizer._prepare_path_tensor(path)  # Prepare tensor for GNN evaluation
                    predicted_cost = self.__queueOptimizer._QueueOptimizer__gnn_model(path_tensor)  # Predict cost

                    if predicted_cost < min_cost:
                        min_cost = predicted_cost
                        best_path = path

                # Assign the best path to the task
                task.setOptimizedPath(best_path)

                # Assign the task to an executor and execute it
                executor = self.__executorsManager.closestFreeExecutor(task)
                if executor is not None:
                    executor.executeJob([self.__queueOptimizer.queue().popTask()])

    # Wait for the queue to be processed (for testing purposes)
    def waitForQueueProcessed(self):
        for i in range(0, 15):
            time.sleep(0.5)
            if self.__started:
                break
        if self.__started:
            while not self.__idle:
                time.sleep(0.5)
            while self.__executorsManager.freeExecutorsNumber() < self.__executorsManager.executorsNumber():
                time.sleep(1)
        else:
            raise Exception("Queue not started yet!")

    # Start queue processing in a separate thread
    def __startQueueProcessingThread(self):
        self.__queueProcessingThread = threading.Thread(target=self.__processQueue)
        self.__queueProcessingThread.daemon = True
        self.__queueProcessingThread.start()

    # Start executors processing in a separate thread
    def __startExecutorsProcessingThread(self):
        self.__executorsManagerThread = threading.Thread(target=self.__processExecutors)
        self.__executorsManagerThread.daemon = True
        self.__executorsManagerThread.start()

    # Start the scheduler (starts queue and executors processing threads)
    def start(self):
        self.__startQueueProcessingThread()
        self.__startExecutorsProcessingThread()

    # Shut down the scheduler
    def shutdown(self):
        self.__killed = True
        if self.__queueProcessingThread:
            self.__queueProcessingThread.join()
            self.__queueProcessingThread = None
        if self.__executorsManagerThread:
            self.__executorsManagerThread.join()
            self.__executorsManagerThread = None
