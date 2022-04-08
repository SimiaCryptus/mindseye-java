/*
 * Copyright (c) 2019 by Andrew Charneski.
 *
 * The author licenses this file to you under the
 * Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance
 * with the License.  You may obtain a copy
 * of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.simiacryptus.mindseye.layers.java;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;
import com.simiacryptus.mindseye.lang.*;
import com.simiacryptus.ref.lang.RefUtil;
import com.simiacryptus.ref.wrappers.RefArrays;
import com.simiacryptus.ref.wrappers.RefIntStream;
import com.simiacryptus.ref.wrappers.RefList;

import javax.annotation.Nonnull;
import java.util.Map;

/**
 * A class to assert the dimensions of a layer.
 *
 * @docgenVersion 9
 */
@SuppressWarnings("serial")
public class AssertDimensionsLayer extends LayerBase {

  private final int[] dims;

  /**
   * Instantiates a new Assert dimensions layer.
   *
   * @param dims the dims
   */
  public AssertDimensionsLayer(final int... dims) {
    super();
    this.dims = dims;
  }

  /**
   * Instantiates a new Assert dimensions layer.
   *
   * @param json the json
   */
  protected AssertDimensionsLayer(@Nonnull final JsonObject json) {
    super(json);
    final JsonArray dimsJson = json.get("dims").getAsJsonArray();
    dims = RefIntStream.range(0, dimsJson.size()).map(i -> dimsJson.get(i).getAsInt()).toArray();
  }

  @Override
  public RefList<Layer> getChildren() {
    return super.getChildren();
  }

  /**
   * Creates an AssertDimensionsLayer from a JSON object.
   *
   * @param json The JSON object to create the layer from.
   * @param rs   A map of character sequences to byte arrays.
   * @return The newly created AssertDimensionsLayer.
   * @docgenVersion 9
   */
  @Nonnull
  @SuppressWarnings("unused")
  public static AssertDimensionsLayer fromJson(@Nonnull final JsonObject json, Map<CharSequence, byte[]> rs) {
    return new AssertDimensionsLayer(json);
  }

  @Override
  public Result eval(@Nonnull final Result... array) {
    if (0 == array.length) {
      RefUtil.freeRef(array);
      throw new IllegalArgumentException(getName());
    }
    Result input = array[0].addRef();
    RefUtil.freeRef(array);
    TensorList inputData = input.getData();
    if (0 == inputData.length()) {
      input.freeRef();
      inputData.freeRef();
      throw new IllegalArgumentException(getName());
    }
    @Nonnull final int[] inputDims = inputData.getDimensions();
    inputData.freeRef();
    if (Tensor.length(inputDims) != Tensor.length(dims)) {
      input.freeRef();
      throw new IllegalArgumentException(
          getName() + ": " + RefArrays.toString(inputDims) + " != " + RefArrays.toString(dims));
    }
    return input;
  }

  @Nonnull
  @Override
  public JsonObject getJson(Map<CharSequence, byte[]> resources, DataSerializer dataSerializer) {
    @Nonnull final JsonObject json = super.getJsonStub();
    @Nonnull final JsonArray dimsJson = new JsonArray();
    for (final int dim : dims) {
      dimsJson.add(new JsonPrimitive(dim));
    }
    json.add("dims", dimsJson);
    return json;
  }

  @Nonnull
  @Override
  public RefList<double[]> state() {
    return RefArrays.asList();
  }

  /**
   * This method frees the object.
   *
   * @docgenVersion 9
   */
  public @SuppressWarnings("unused")
  void _free() {
    super._free();
  }

  @Nonnull
  public @Override
  @SuppressWarnings("unused")
  AssertDimensionsLayer addRef() {
    return (AssertDimensionsLayer) super.addRef();
  }

}
