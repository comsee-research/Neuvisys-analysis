#include "libserie/Faulhaber.hpp"
#include <iostream>
#include <vector>
#include <map>

void testMotor() {
    std::string port = "/dev/ttyUSB0";
    std::string position_finale;

    std::vector<int> position;
    for (int i = 0; i < 300000; i += 10000) {
        position.push_back(i);
    }
    for (int i = 300000; i >= 0; i -= 10000) {
        position.push_back(i);
    }

    Faulhaber Motor1(1, port);
    Motor1.StartDrive();

    for (auto pos : position) {
        Motor1.SetAbsolutePosition(pos);
    }

    Motor1.StopDrive();
}

int main(int argc, char* argv[]) {
    std::vector<std::string> args(argv, argv + argc); 
    std::string port="/dev/ttyUSB0";
    std::string position_finale;

    if (argc == 1) {
        std::cout << "Pas assez d'arguments. Lancez GOTO num_moteur position [num_moteur position ...] [port]" << std::endl;
        testMotor();
        return 0;
    }
    if (!(argc % 2)) {
        std::cout << "port : " << args[argc-1] << std::endl;
        port = args[argc-1];
        args.pop_back();
    }
    std::map<std::string, std::string> Commande;

    for(auto it = args.begin()+1; it != args.end(); it+=2) {
        Commande[*it] =* (it+1);
    }

    for (const auto& arg : Commande) {
        std:: cout << " Moteur " << arg.first << " goto " << arg.second << "." << std::endl;
        Faulhaber Servo(std::stoi(arg.first), port);
        Servo.StartDrive();
        Servo.SetAbsolutePosition(std::stoi(arg.second));
        Servo.GetPosition(position_finale);
        Servo.StopDrive();
    }
}
