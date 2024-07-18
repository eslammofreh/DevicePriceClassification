package com.DevicePriceClassification.DevicePriceClassification.model;

import jakarta.persistence.*;
import lombok.Data;

@Data // Lombok annotation to generate getters, setters, toString, etc.
@Entity // JPA annotation to make this class a JPA entity
@Table(name = "devices") // Specify the table name in the database
public class Device {

    @Id // Marks this field as the primary key
    @GeneratedValue(strategy = GenerationType.IDENTITY) // Auto-generate the ID
    private Long id;

    private int battery_power; // Total energy a battery can store in one time measured in mAh
    private int blue; // Has Bluetooth or not
    private double clock_speed; // The speed at which the microprocessor executes instructions
    private int dual_sim; // Has dual sim support or not
    private int fc; // Front Camera megapixels
    private int four_g; // Has 4G or not
    private int int_memory; // Internal Memory in Gigabytes
    private double m_dep; // Mobile Depth in cm
    private int mobile_wt; // Weight of mobile phone
    private int n_cores; // Number of cores of the processor
    private int pc; // Primary Camera megapixels
    private int px_height; // Pixel Resolution Height
    private int px_width; // Pixel Resolution Width
    private int ram; // Random Access Memory in Megabytes
    private double sc_h; // Screen Height of mobile in cm
    private double sc_w; // Screen Width of mobile in cm
    private int talk_time; // Longest time that a single battery charge will last
    private int three_g; // Has 3G or not
    private int touch_screen; // Has touch screen or not
    private int wifi; // Has wifi or not

    private Integer price_range; // Predicted price range (0: low cost, 1: medium cost, 2: high cost, 3: very high cost)
}