__host__ void logging(std::string file_name, int step,
double *d_mdVx , double *d_mdVy , double *d_mdVz,
double *d_vx , double *d_vy , double *d_vz, int N , int Nmd, int grid_size, double *temperature)
{   
        std::ofstream log (file_name, std::ios_base::app); 
        //help variable:
        double *d_tmp;
        double E,px,py,pz;
        cudaMalloc((void**)&d_tmp, sizeof(double)*grid_size);

        sumCommMultiBlock<<<grid_size, blockSize>>>(d_vx, N, d_tmp);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
        sumCommMultiBlock<<<1, blockSize>>>(d_tmp, grid_size, d_tmp);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
        cudaMemcpy(&px, d_tmp, sizeof(double), cudaMemcpyDeviceToHost);
        sumCommMultiBlock<<<grid_size, blockSize>>>(d_vy, N, d_tmp);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
        sumCommMultiBlock<<<1, blockSize>>>(d_tmp, grid_size, d_tmp);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
        cudaMemcpy(&py, d_tmp, sizeof(double), cudaMemcpyDeviceToHost);
        sumCommMultiBlock<<<grid_size, blockSize>>>(d_vz, N, d_tmp);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
        sumCommMultiBlock<<<1, blockSize>>>(d_tmp, grid_size, d_tmp);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
        cudaMemcpy(&pz, d_tmp, sizeof(double), cudaMemcpyDeviceToHost);
        sumsquared3arrayCommMultiBlock<<<grid_size, blockSize>>>(d_vx, d_vy, d_vz, N, d_tmp);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
        sumCommMultiBlock<<<1, blockSize>>>(d_tmp, grid_size, d_tmp);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
        cudaMemcpy(&E, d_tmp, sizeof(double), cudaMemcpyDeviceToHost);
        double *h_mdVx , *h_mdVy , *h_mdVz;
        h_mdVx = (double*)malloc(sizeof(double) * Nmd); h_mdVy = (double*)malloc(sizeof(double) * Nmd); h_mdVz = (double*)malloc(sizeof(double) * Nmd);
        cudaMemcpy(h_mdVx, d_mdVx, sizeof(double) * Nmd , cudaMemcpyDeviceToHost);
        cudaMemcpy(h_mdVy, d_mdVy, sizeof(double) * Nmd , cudaMemcpyDeviceToHost);
        cudaMemcpy(h_mdVz, d_mdVz, sizeof(double) * Nmd , cudaMemcpyDeviceToHost); 

        double md_p_x = momentum(density, h_mdVx, Nmd);
        double md_p_y = momentum(density, h_mdVy, Nmd);
        double md_p_z = momentum(density, h_mdVz, Nmd);

        log<<"Step:"<<step<<"\nTemp = "<<temp_calc(d_vx, d_vy , d_vz , d_mdVx , d_mdVy , d_mdVz , density, N , Nmd, grid_size)<<", <p_x> = "<<
        (px+md_p_x)/(N+Nmd*density) <<", <p_y> = "<<(py+md_p_y)/(N+Nmd*density)<<
        ", <p_z> = "<<(pz+md_p_z)/(N+Nmd*density)<<"\n";
        
        //*temperature = temp_calc(d_vx, d_vy , d_vz , d_mdVx , d_mdVy , d_mdVz , density, N , Nmd, grid_size);
        printf("Step:%i\nTemp = %f, <p_x> = %f, <p_y> = %f, <p_z> = %f\n",step,temp_calc(d_vx, d_vy , d_vz , d_mdVx , d_mdVy , d_mdVz , density, N , Nmd, grid_size), (px+md_p_x)/(N+Nmd*density) ,(py+md_p_y)/(N+Nmd*density) ,(pz+md_p_z)/(N+Nmd*density));
}

__host__ void xyz_trj(std::string file_name,  double *d_mdX, double *d_mdY , double *d_mdZ, int Nmd)
{
    std::ofstream traj (file_name, std::ios_base::app);
    double *h_mdX, *h_mdY, *h_mdZ;
    h_mdX = (double*)malloc(sizeof(double) * Nmd);  
    h_mdY = (double*)malloc(sizeof(double) * Nmd);  
    h_mdZ = (double*)malloc(sizeof(double) * Nmd);
    cudaMemcpy(h_mdX, d_mdX, sizeof(double) * Nmd , cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mdY, d_mdY, sizeof(double) * Nmd , cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mdZ, d_mdZ, sizeof(double) * Nmd , cudaMemcpyDeviceToHost);
    traj<<Nmd<<"\n\n";
    for (int i =0 ; i< Nmd ; i++)
    {
        traj<<"C      "<<h_mdX[i]<<"      "<<h_mdY[i]<<"      "<<h_mdZ[i]<<"\n";
    }

}

//this function is the same as xyz_trj, but in this version the origin (0 , 0 , 0) is ommited and won't be saved.
__host__ void xyz_trj_mpcd(std::string file_name,  double *d_X, double *d_Y , double *d_Z, int Nmd , int zerofactor_sum)
{
    std::ofstream traj (file_name, std::ios_base::app);

    int N_nonzero = Nmd - zerofactor_sum;
    double *h_X, *h_Y, *h_Z;
    h_X = (double*)malloc(sizeof(double) * Nmd);  
    h_Y = (double*)malloc(sizeof(double) * Nmd);  
    h_Z = (double*)malloc(sizeof(double) * Nmd);
    cudaMemcpy(h_X, d_X, sizeof(double) * Nmd , cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Y, d_Y, sizeof(double) * Nmd , cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Z, d_Z, sizeof(double) * Nmd , cudaMemcpyDeviceToHost);
    //printf("N_nonzero=%i\n", N_nonzero);
    int counter = 0;
    int zerocounter = 0;
    //traj<<N_nonzero<<"\n\n";
    traj<<"N      "<<N_nonzero<<"      "<<0<<"      "<<0<<"\n";
    //traj<<Nmd<<"\n\n";
    for (int i =0 ; i< Nmd ; i++)
    {
        if (h_X[i] == 1000.0000000 && h_Y[i] == 1000.0000000 && h_Z[i] == 1000.0000000) {
            //printf("zerooo\n");
            zerocounter = zerocounter + 1;
        }

        else if (h_X[i] != 1000.0000000 || h_Y[i] != 1000.0000000 || h_Z[i] != 1000.0000000){
            traj<<"C      "<<h_X[i]<<"      "<<h_Y[i]<<"      "<<h_Z[i]<<"\n";
            
            counter = counter +1;
            //printf("nonzero\n");
        }
    }
    //printf("counter = %i\n", counter);
    //printf("zerocounter = %i\n", zerocounter);
    

}

__host__ void zero_factor(std::string file_name,int Nmd, int zerofactor_sum){

    std::ofstream account(file_name, std::ios_base::app);
    int N_nonzero = Nmd - zerofactor_sum;
    account<<N_nonzero<<"\n";
}

void flowprofile(int *d_index,double *d_vx, FILE *file, int N)
    {
        double *vx;
        int *index;
        index = (int*)malloc(sizeof(int) * N);  
        vx = (double*)malloc(sizeof(double) * N);
        cudaMemcpy(index, d_index, sizeof(int) * N , cudaMemcpyDeviceToHost);
        cudaMemcpy(vx, d_vx, sizeof(double) * N , cudaMemcpyDeviceToHost); 
        double flow[16]={0.};
        int num[16]={0};
        //std::cout<<"fucking here"<<std::endl;
        //usleep(3 * microsecond);//sleeps for 3 second
        for (unsigned int i =0; i<N ; i++)
        {
            //std::cout<<"fucking here"<<i<<std::endl;
            //usleep(3 * microsecond);//sleeps for 3 second
            int a =int(index[i]/16/16);
            if(a<0)
            {
                std::cout<<"index found negetive!"<<std::endl;
            }
            if(a>16)
            {
                std::cout<<a<<std::endl;
            }
            flow[a] +=vx[i];
            num[a]++;
            
            //std::cout<<index[i]<<"\n";
        }

        for (unsigned int i =0 ; i<16; i++)
            
            fprintf(file, "%f,", flow[i]/num[i]);
        fprintf(file,"\n");
        free(vx);
        free(index);
    }


__host__ void xyz_trjvel(std::string file_name,  double *d_X, double *d_Y , double *d_Z, double *d_Vx, double *d_Vy, double *d_Vz, int NN)
{
    std::ofstream trajvel (file_name, std::ios_base::app);
    double *h_X, *h_Y, *h_Z;
    h_X = (double*)malloc(sizeof(double) * NN);  
    h_Y = (double*)malloc(sizeof(double) * NN);  
    h_Z = (double*)malloc(sizeof(double) * NN);
    double *h_Vx, *h_Vy, *h_Vz;
    h_Vx = (double*)malloc(sizeof(double) * NN);
    h_Vy = (double*)malloc(sizeof(double) * NN);
    h_Vz = (double*)malloc(sizeof(double) * NN);
    cudaMemcpy(h_X, d_X, sizeof(double) * NN , cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Y, d_Y, sizeof(double) * NN , cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Z, d_Z, sizeof(double) * NN , cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Vx, d_Vx, sizeof(double) * NN , cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Vy, d_Vy, sizeof(double) * NN , cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Vz, d_Vz, sizeof(double) * NN , cudaMemcpyDeviceToHost);
    trajvel<<NN<<"\n\n";
    for (int i =0 ; i< NN ; i++)
    {
        trajvel<<"C      "<<h_X[i]<<"      "<<h_Y[i]<<"      "<<h_Z[i]<<"       "<<h_Vx[i]<<"       "<<h_Vy[i]<<"       "<<h_Vz[i]<<"\n";
    }

}