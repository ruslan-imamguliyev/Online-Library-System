package com.project.e_library.entity;

import jakarta.persistence.*;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.Setter;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

@Getter
@Setter(AccessLevel.NONE)
@Entity
@Table(name = "library")
public class Book {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    @Column(name = "id")
    private int id;

    @Column(name = "title")
    private String title;

    @Column(name = "author")
    private String author;

    @Column(name = "genre")
    private String genre;

    @Column(name = "description")
    private String description;

    @Column(name = "coverImg")
    private String imgUrl;

    @Column(name = "likedPercent")
    private double ratingPercent;

    private double rating;

    @Column(name = "numRatings")
    private long ratingNumber;


    @Transient
    public List<String> getGenres() {
        if (genre == null) return List.of();
        String genresStr = genre.replace("[", "")
                .replace("]", "")
                .replace("'", "")
                .replace("\"", "");

        String[] genreArray = genresStr.split(",");
        return new ArrayList<>(Arrays.asList(genreArray));
    }

}
