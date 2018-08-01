
package edu.cmu.cs.gabriel.network;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.net.UnknownHostException;
import java.util.Timer;
import java.util.TimerTask;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Handler;
import android.os.Message;
import android.util.Base64;
import android.util.Log;
import edu.cmu.cs.gabriel.token.ReceivedPacketInfo;

public class ResultReceivingThread extends Thread {

    private static final String LOG_TAG = "ResultThread";

    private boolean isRunning = false;

    // TCP connection
    private InetAddress remoteIP;
    private int remotePort;
    private Socket tcpSocket;
    private DataOutputStream networkWriter;
    private DataInputStream networkReader;

    private Handler returnMsgHandler;


    public ResultReceivingThread(String serverIP, int port, Handler returnMsgHandler) {
        isRunning = false;
        this.returnMsgHandler = returnMsgHandler;
        try {
            remoteIP = InetAddress.getByName(serverIP);
        } catch (UnknownHostException e) {
            Log.e(LOG_TAG, "unknown host: " + e.getMessage());
        }
        remotePort = port;
    }

    @Override
    public void run() {
        this.isRunning = true;
        Log.i(LOG_TAG, "Result receiving thread running");

        try {
            tcpSocket = new Socket();
            tcpSocket.setTcpNoDelay(true);
            tcpSocket.connect(new InetSocketAddress(remoteIP, remotePort), 5*1000);
            networkWriter = new DataOutputStream(tcpSocket.getOutputStream());
            networkReader = new DataInputStream(tcpSocket.getInputStream());
        } catch (IOException e) {
            Log.e(LOG_TAG, "Error in initializing Data socket: " + e);
            this.notifyError(e.getMessage());
            this.isRunning = false;
            return;
        }

        while (isRunning == true){
            try {
                String recvMsg = this.receiveMsg(networkReader);
                this.notifyReceivedData(recvMsg);
            } catch (IOException e) {
                Log.w(LOG_TAG, "Error in receiving result, maybe because the app has paused");
                this.notifyError(e.getMessage());
                break;
            }
        }
    }

    /**
     * @return a String representing the received message from @reader
     */
    private String receiveMsg(DataInputStream reader) throws IOException {
        int retLength = reader.readInt();
        byte[] recvByte = new byte[retLength];
        int readSize = 0;
        while(readSize < retLength){
            int ret = reader.read(recvByte, readSize, retLength-readSize);
            if(ret <= 0){
                break;
            }
            readSize += ret;
        }
        String receivedString = new String(recvByte);
        return receivedString;
    }


    private void notifyReceivedData(String recvData) {
        // convert the message to JSON
        String status = null;
        String result = null;
        String sensorType = null;
        long frameID = -1;
        String engineID = "";
        int injectedToken = 0;

        try {
            JSONObject recvJSON = new JSONObject(recvData);
            status = recvJSON.getString("status");
            result = recvJSON.getString(NetworkProtocol.HEADER_MESSAGE_RESULT);
            sensorType = recvJSON.getString(NetworkProtocol.SENSOR_TYPE_KEY);
            frameID = recvJSON.getLong(NetworkProtocol.HEADER_MESSAGE_FRAME_ID);
            engineID = recvJSON.getString(NetworkProtocol.HEADER_MESSAGE_ENGINE_ID);
            //injectedToken = recvJSON.getInt(NetworkProtocol.HEADER_MESSAGE_INJECT_TOKEN);
        } catch (JSONException e) {
            Log.e(LOG_TAG, recvData);
            Log.e(LOG_TAG, "the return message has no status field");
            return;
        }


        // return status
        if (sensorType.equals(NetworkProtocol.SENSOR_JPEG)) {
            Message msg = Message.obtain();
            msg.what = NetworkProtocol.NETWORK_RET_MESSAGE;
            msg.obj = new ReceivedPacketInfo(frameID, engineID, status);
            this.returnMsgHandler.sendMessage(msg);
        }

        if (!status.equals("success")) {
            if (sensorType.equals(NetworkProtocol.SENSOR_JPEG)) {
                Message msg = Message.obtain();
                msg.what = NetworkProtocol.NETWORK_RET_DONE;
                this.returnMsgHandler.sendMessage(msg);
            }
            return;
        }

        // TODO: refilling tokens
//        if (injectedToken > 0){
//            this.tokenController.increaseTokens(injectedToken);
//        }

        if (result != null){
            /* parsing result */
            JSONObject resultJSON = null;
            try {
                resultJSON = new JSONObject(result);
            } catch (JSONException e) {
                Log.e(LOG_TAG, "Result message not in correct JSON format");
            }

            String speechFeedback = "";
            Bitmap imageFeedback = null;

            // image guidance
            try {
                String imageFeedbackString = resultJSON.getString("annotated_img");
                byte[] data = Base64.decode(imageFeedbackString.getBytes(), Base64.DEFAULT);
                imageFeedback = BitmapFactory.decodeByteArray(data, 0, data.length);
                //Log.e(LOG_TAG, "" + imageFeedback.getHeight() + "," + imageFeedback.getWidth());

                Message msg = Message.obtain();
                msg.what = NetworkProtocol.NETWORK_RET_IMAGE;
                msg.obj = imageFeedback;
                this.returnMsgHandler.sendMessage(msg);
            } catch (JSONException e) {
                Log.v(LOG_TAG, "no image annotation found");
            }

            // text guidance
            try {
                String textFeedback = resultJSON.getString("annotated_text");
                Message msg = Message.obtain();
                msg.what = NetworkProtocol.NETWORK_RET_TEXT;
                msg.obj = textFeedback;
                //this.returnMsgHandler.sendMessage(msg);
            } catch (JSONException e) {
                Log.v(LOG_TAG, "no text annotation found");
            }

            // speech guidance
            try {
                speechFeedback = resultJSON.getString("speech");
                Message msg = Message.obtain();
                msg.what = NetworkProtocol.NETWORK_RET_SPEECH;
                msg.obj = speechFeedback;
                this.returnMsgHandler.sendMessage(msg);
            } catch (JSONException e) {
                Log.v(LOG_TAG, "no speech guidance found");
            }

            // done processing return message
            if (sensorType.equals(NetworkProtocol.SENSOR_JPEG)) {
                Message msg = Message.obtain();
                msg.what = NetworkProtocol.NETWORK_RET_DONE;
                this.returnMsgHandler.sendMessage(msg);
            }
        }
    }

    public void close() {
        this.isRunning = false;

        try {
            if(this.networkReader != null){
                this.networkReader.close();
                this.networkReader = null;
            }
        } catch (IOException e) {
        }
        try {
            if(this.networkWriter != null){
                this.networkWriter.close();
                this.networkWriter = null;
            }
        } catch (IOException e) {
        }
        try {
            if(this.tcpSocket != null){
                this.tcpSocket.shutdownInput();
                this.tcpSocket.shutdownOutput();
                this.tcpSocket.close();
                this.tcpSocket = null;
            }
        } catch (IOException e) {
        }
    }

    /**
     * Notifies error to the main thread
     */
    private void notifyError(String errorMessage) {
        Message msg = Message.obtain();
        msg.what = NetworkProtocol.NETWORK_RET_FAILED;
        msg.obj = errorMessage;
        this.returnMsgHandler.sendMessage(msg);
    }
}
