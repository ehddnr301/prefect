import prefect
from prefect import Flow
from prefect.schedules.schedules import CronSchedule
from task.prep import *


class Pipeline:
    _project_name = None
    _flow_name = None
    _logger = None
    _flow = None

    '''
        _param1 = Parameter("data_path", default="default_path")
        _param2 = Parameter("model_name", default="GPN")
    '''

    def __init__(self, project_name, flow_name, schedule=None):
        self._logger = prefect.context.get("logger")
        self._logger.info("Create Pipeline")

        self._project_name = project_name
        self._flow_name = flow_name
        self._schedule = schedule

    def create_flow(self):
        self._logger.info(f"Create {self._flow_name} flow")
        with Flow(self._flow_name) as flow:
            """

            data = load_data(self._param1)
            prep_data = preprocess(data)
            model = train(self._param2, prep_data)
            save_model(model)
            
            """
            # name = hello_task("kyle")
            # name1 = hi_task(name)
            # name2 = hi_task(name)
            # buy_task(name1 + name2)
            # buy_task(name1 + name2)

            DF_PATH = './train.csv'
            BATCH_SIZE = 64
            TRAINING_EPOCH=3
            DEVICE = 'cpu'
            CRITERION = torch.nn.CrossEntropyLoss().to(DEVICE)
            LEARNING_RATE = 1e-3
            train_transform = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=(0.5,), std=(1.0,))
                        ])
            Net = MnistNet()
            optimizer = torch.optim.Adam(Net.parameters(), lr=LEARNING_RATE)

            train_df, valid_df = load_dataset(df_path=DF_PATH)
            train_loader, valid_loader, total_batch = preprocess_train(Dataset=Dataset, transform=train_transform, batch_size=BATCH_SIZE, train_df=train_df, valid_df=valid_df)
            train_Net = cnn_training(Net=Net, train_loader=train_loader, epoch=TRAINING_EPOCH, device=DEVICE, criterion=CRITERION, optimizer=optimizer, total_batch=total_batch)
            save_model()
            Net2 = return_Net2(train_Net)
            feature_weight_df = make_knn_feature(train_df=train_df, train_loader=train_loader, device=DEVICE, optimizer=optimizer, Net2=Net2)
            KNN = knn_training(neighbors=3, feature_weight_df=feature_weight_df)
            knn_result = predict_knn_model(323, train_df=train_df, valid_df=valid_df, transform=train_transform, Net2=Net2, KNN=KNN)
                                            
            # implement here
            # ...

        self._flow = flow
        self._register()

    def _register(self):
        self._logger.info(
            f"Regist {self._flow_name} flow to {self._project_name} project"
        )
        self._logger.info(f"Set Cron {self._schedule}")

        self._flow.register(
            project_name=self._project_name, 
            idempotency_key=self.flow.serialized_hash()
        )

        if self._schedule:
            self._set_cron()

    def _set_cron(self):
        self.flow.schedule(CronSchedule(self._schedule))

    @property
    def flow(self):
        return self._flow

    @property
    def project_name(self):
        return self._project_name

    @property
    def flow_name(self):
        return self._flow_name
