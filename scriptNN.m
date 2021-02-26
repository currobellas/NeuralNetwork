
% Script to get the prediction modle with NN. Required init correctly
% the next variables:
% trainData: Train variables array
% trainLabels: Train output vector
% testData: Validation variables array
% testLabels: Validation output vector

listaIteraciones=[50, 100, 500, 1000, 1500, 2000];
lambdas=[0.00001,0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, ...
    0.1, 0.5, 1];
nombreBase='ModeloA_NN'; % Initial desired file name
iteraciones=5000;
for ocultas=1:10
    fprintf('\nComenzando con %d capas ocultas',ocultas);
    for lambdaIndex=1:length(lambdas)
        lambda=lambdas(lambdaIndex);
        fprintf('\n\tlamda %f.',lambda);
        for indiceIteraciones=1:length(listaIteraciones)
            iteraciones=listaIteraciones(indiceIteraciones);
            
            tic;
            X=trainData;     
            m = size(X, 1); % # train samples
            nIn= size(X,2); % # input data
            y=(trainLabels+1)/2; % Output labels (0 and  1)
          
            
            % Input layer
            input_layer_size  = length( trainData(1,:));
            %  Hidden layers
            hidden_layer_size = ocultas;
            % Output 1
            num_labels = 1;
            
            initial_Theta1 = ...
                randInitializeWeights(input_layer_size, hidden_layer_size);
            initial_Theta2 = ...
                randInitializeWeights(hidden_layer_size, num_labels);
            
            % Unroll parameters
            initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
            
            % Create result file
            fid=fopen(strcat(nombreBase,'_reg',num2str(lambda), ...
                '_iter',num2str(iteraciones), '_capas',num2str(ocultas),...
                '.txt'), 'wt'); %w:write (para append a+) t:text
            
            %  Iterations 
            options = optimset('MaxIter', iteraciones);
            
            % NN Cost function 
            costFunction = @(p) nnCostFunction(p, ...
                input_layer_size, ...
                hidden_layer_size, ...
                num_labels, X, y, lambda);
            
            % Minimize cost function with 
            % fmincg (Copyright (C) 2001 and 2002 by Carl
            % Edward Rasmussen. Date 2002-02-13)
            [nn_params, cost] = fmincg(costFunction, initial_nn_params, ...
                options);
            
            %  theta1 and theta2
            Theta1 = reshape(nn_params(1:hidden_layer_size * ...
                (input_layer_size + 1)), ...
                hidden_layer_size, (input_layer_size + 1));
            
            Theta2 = reshape(nn_params((1 + (hidden_layer_size * ...
                (input_layer_size + 1))):end), ...
                num_labels, (hidden_layer_size + 1));
            
            
            % Prediction with trainData and testData.
            pred = predict(Theta1, Theta2, X);
            predTest = predict(Theta1, Theta2, testData);
            
            salidaTrain=pred*2-1;
            salidaTest=predTest*2-1;
            
            % Predict with test set in first loop
            % and train set in second loop
            X=testData;
            Y=testLabels;
            salida = salidaTest;
            mensajePrediccion='\nPredicción con conjunto de testeo\n';
            for i=1:2
                toc;
                fprintf(fid,'\n-----------------------\n');
                fprintf(fid,mensajePrediccion);
                fprintf(fid, 'Instante: %f\n',toc);
                fprintf(fid, 'Lambda: %f\n', lambda);
                fprintf(fid, 'Capas ocultas: %d\n', ocultas);
                fprintf(fid, 'Iteraciones: %d\n', iteraciones);
                
                % Confusion matrix
                Cmas1=sum((salida==1) & (Y==1));
                Mmas1=sum((salida==-1) & (Y==1));
                Mmenos1=sum((salida==1) & (Y==-1));
                Cmenos1=sum((salida==-1) & (Y==-1));
                fprintf(fid, '\nMatriz de Confusion:\n');
                fprintf(fid, 'C+1: %4d\tM+1: %4d\nM-1: %4d\tC-1: %4d', ...
                    Cmas1, Mmas1, Mmenos1, Cmenos1);
                
                % Statistics
                NT=Cmas1+Cmenos1+Mmas1+Mmenos1;
                Pg=(Cmas1+Cmenos1)/NT;
                PG=Pg*100;
                Pa=((Cmas1*Cmenos1)+(Mmas1*Mmenos1))/(NT^2);
                Kappa=100*(Pg-Pa)/(1-Pa);
                Recall=Cmas1/(Cmas1+Mmas1);
                Precision=Cmas1/(Cmas1+Mmenos1);
                F1=2*(Recall*Precision)/(Recall+Precision);
                Specificity=Cmenos1/(Cmenos1+Mmenos1);
                
                fprintf(fid, ...
                    '\nEstadisticos finales:\n\tPG: %f\n\tKappa: %f\n', ...
                    PG, Kappa);
                fprintf(fid, ...
                    '\tRecall: %f\n\tPrecision: %f\n\tF1Score: %f\n', ...
                    Recall, Precision, F1);
                fprintf(fid,' \tSpecificity: %f\n', Specificity);
                
                % Change to train set
                X=trainData;
                Y=trainLabels;
                salida=salidaTrain;
                mensajePrediccion='\nPrediccion con conjunto de entrenamiento\n';
            end
            fclose(fid);
        end
    end
end

