#include <unistd.h>
#include <string>
#include "libserie/comserie.hpp"

using namespace std;

//==============================================================================

int main(int argc, char **argv) {
    ComSerie com;
    uint8_t octet = 0;
    string chaine = "";

    //trames de configuration �transmettre

    com.ouvrir("/dev/ttyUSB0", B9600 | CS8 | CLOCAL | CREAD, 10);
    chaine += "0GSER\r\n";; //Get Serial 0
    chaine += "0EN\r\n";  // Enable drive 0
    chaine += "0CST\r\n";  //Set operating mode
    com.envoyer((const uint8_t *) (chaine.c_str()), uint32_t(chaine.length()));
    com.flush();
    chaine = "";


    chaine += "1GSER\r\n";; //Get Serial 0
    chaine += "1EN\r\n";  // Enable drive 0
    chaine += "1CST\r\n";  //Set operating mode
    com.envoyer((const uint8_t *) (chaine.c_str()), uint32_t(chaine.length()));
    com.flush();
    chaine = "";

    //GOTO 1ST POSITION
    chaine += "0LA-120000\r\n"; // Load Absolute position to go
    com.envoyer((const uint8_t *) (chaine.c_str()), uint32_t(chaine.length()));
    com.flush();
    chaine = "";
    chaine += "0NP\r\n";  // Stop on the command M until the position is reached
    chaine += "0M\r\n";
    com.envoyer((const uint8_t *) (chaine.c_str()), uint32_t(chaine.length()));
    com.flush();
    chaine = "";
    sleep(3);

    chaine += "1LA-120000\r\n"; // Load Absolute position to go
    com.envoyer((const uint8_t *) (chaine.c_str()), uint32_t(chaine.length()));
    com.flush();
    chaine = "";
    chaine += "1NP\r\n";  // Stop on the command M until the position is reached
    chaine += "1M\r\n";
    com.envoyer((const uint8_t *) (chaine.c_str()), uint32_t(chaine.length()));
    com.flush();
    chaine = "";
    sleep(3);

    //GOTO 2d POSITION
    chaine = "";
    chaine += "0LA50000\r\n"; // Load Absolute position to go
    com.envoyer((const uint8_t *) (chaine.c_str()), uint32_t(chaine.length()));
    com.flush();
    chaine = "";
    chaine += "0NP\r\n";  // Stop on the command M until the position is reached
    chaine += "0M\r\n";
    com.envoyer((const uint8_t *) (chaine.c_str()), uint32_t(chaine.length()));
    com.flush();
    chaine = "";

    chaine = "";
    chaine += "1LA50000\r\n"; // Load Absolute position to go
    com.envoyer((const uint8_t *) (chaine.c_str()), uint32_t(chaine.length()));
    com.flush();
    chaine = "";
    chaine += "1NP\r\n";  // Stop on the command M until the position is reached
    chaine += "1M\r\n";
    com.envoyer((const uint8_t *) (chaine.c_str()), uint32_t(chaine.length()));
    com.flush();
    chaine = "";

    //STOP
    sleep(10);
    chaine += "0DI\r\n"; //Disable drive
    chaine += "0CST\r\n";
    chaine += "0POS\r\n"; //Get actual position*/

    chaine += "1DI\r\n"; //Disable drive
    chaine += "1CST\r\n";
    chaine += "1POS\r\n"; //Get actual position*/

    com.envoyer((const uint8_t *) (chaine.c_str()), uint32_t(chaine.length()));
    com.flush();
    chaine = "";
//com.ouvrir("/dev/ttyS0",B9600|CRTSCTS|CS8|CLOCAL|CREAD,10);
    //com.ouvrir("/dev/ttyS0",B19200|CRTSCTS|CS8|CLOCAL|CREAD,10);
    com.ouvrir("/dev/ttyUSB0", B9600 | CS8 | CLOCAL | CREAD, 10);
    //com.envoyer((const uint8_t *)(chaine.c_str()),uint32_t(chaine.length()));
    //com.flush();
//sleep(1);
    std::cout << " Sent :: " << std::endl;
    int i = 0;
    do {
        i++;
        if (com.recevoir(&octet, 1)) {
            cout << octet;
        }
    } while (i < 7);


    com.flush();

    //affichage des 100 trames suivantes recues
    //(boucle infinie si non NMEA)
    int nbTrames = 0;
    while (nbTrames <= 100) {
        if (com.recevoir(&octet, 1)) {
            if (octet == '\n') ++nbTrames;
            cout << octet;
        } else cout << "timeout" << endl;
    }

    com.fermer();

    return 0;
}
//==============================================================================
