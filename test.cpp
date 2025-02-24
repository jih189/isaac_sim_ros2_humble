__global__ void kernelA() {
    // do something
}

__global__ void kernelB() {
    // do something
}

__global__ void kernelC() {
    // do something
}

__global__ void kernelD() {
    // do something
}

// base class
class Base {
public:
    virtual void function() = 0;
};


// derived class A
class A : public Base {
public:
    void function() {
        kernelA();
        cudaDeviceSynchronize();
        kernelC();
        cudaDeviceSynchronize();
    }
};

// derived class B
class B : public Base {
public:
    void function() {
        kernelB();
        cudaDeviceSynchronize();
        kernelD();
        cudaDeviceSynchronize();
    }
};

int main() {
    std::vector<Base*> objects;
    objects.push_back(new A());
    objects.push_back(new B());

    for (auto obj : objects) {
        obj->function();
    }
}